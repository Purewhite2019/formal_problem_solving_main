import time
import traceback
from typing import Any, Optional

from loguru import logger
from openai import NOT_GIVEN
from transformers import AutoTokenizer

from agent.base import ProofAttempt, ProofResult
from agent.whole_proof_generation import (
    LEAN4_DEFAULT_IMPORTS,
    OPTION_HEADER,
    WholeProofGenerationAgent,
    remove_min_indent_until_fence,
)
from common.constants import BANNED_TOKENS
from common.pantograph.dataclasses import FormalProblem, GoalState
from common.utils import remove_comments


class WholeProofGenerationCoTAgent(WholeProofGenerationAgent):
    token_limit = 8192
    stop_token = "```lean4"

    def __init__(self, gen_client, gen_model_name, num_samples_per_trial=128,
                 temperature=1.0, top_p=0.95, max_tokens=2048, **kwargs):
        super().__init__(gen_client, gen_model_name, num_samples_per_trial,
                         temperature, top_p, max_tokens)
        self._tokenizer = None

    def _get_tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.gen_model_name, trust_remote_code=True)
        return self._tokenizer

    def _format_prompt(self, sample: FormalProblem, formal_prefix: str) -> str:
        tokenizer = self._get_tokenizer()
        user_content = self.build_prompt(sample, formal_prefix)
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)

    def completion_block(self, formal_prefix: str) -> str:
        return f"```lean4\n{formal_prefix}"

    def post_process(self, output: str) -> str:
        return remove_comments(remove_min_indent_until_fence(output))

    async def prove_async(self, sample, server, init_state, tag="", verbose=False):
        start_time = time.time()
        formal_prefix = server.formal_problem_framework_prefix
        initial_prompt = self._format_prompt(sample, formal_prefix)
        attempts = []
        final_state = None
        final_proof = None
        log_debug = logger.info if verbose else logger.debug
        tokenizer = self._get_tokenizer()

        for sample_i in range(self.num_samples_per_trial):
            attempt_start = time.time()
            try:
                prompt = initial_prompt
                text_block = ""
                while True:
                    prompt_len = len(tokenizer.tokenize(prompt))
                    assert prompt_len < self.token_limit, f"Current prompt length: {prompt_len}"
                    response = await self.gen_client.completions.create(
                        model=self.gen_model_name,
                        prompt=prompt,
                        max_tokens=self.token_limit - prompt_len,
                        stream=False,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        n=1,
                        stop=self.stop_token,
                        extra_headers={"include_stop_str_in_output": "true"},
                    )
                    text_block = response.choices[0].text
                    try:
                        stop_reason = response.choices[0].stop_reason
                    except AttributeError:
                        stop_reason = response.choices[0].finish_reason

                    if stop_reason == self.stop_token:
                        completion_block = self.completion_block(formal_prefix)
                        prompt = prompt + text_block + completion_block
                    else:
                        proof = "{\n" + self.post_process(text_block) + "\n}"
                        break

                assert all(token not in proof for token in BANNED_TOKENS), \
                    f"Banned token detected in proof: {proof}"

                state = await server.server.goal_tactic_async(init_state, 0, proof)
                success = state.is_solved
                attempts.append(ProofAttempt(
                    success=success,
                    duration=time.time() - attempt_start,
                    proof=proof,
                    response_content=text_block,
                    final_state=state,
                ))
                if success:
                    final_state = state
                    final_proof = proof
                    break
            except Exception as exc:
                error = traceback.format_exc()
                attempts.append(ProofAttempt(
                    success=False,
                    duration=time.time() - attempt_start,
                    error=error))
                log_debug(f"prove({tag}, {sample_i}/{self.num_samples_per_trial}) failed: {repr(exc)}")

        success = final_state is not None and final_state.is_solved
        return ProofResult(
            success=success,
            duration=time.time() - start_time,
            proof=[(0, final_proof)] if success else [],
            final_state=final_state,
            attempts=attempts,
        )


class DeepSeekProverV2CoTAgent(WholeProofGenerationCoTAgent):
    agent_name = "deepseek_prover_v2_cot"
    token_limit = 8191

    prompt_template = (
        "Complete the following Lean 4 code:\n\n"
        "```lean4\n{}\n```\n\n"
        "Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.\n"
        "The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."
    )

    def build_prompt(self, sample: FormalProblem, formal_prefix: str) -> str:
        composed = (f"{LEAN4_DEFAULT_IMPORTS}\n\n{OPTION_HEADER}\n\n"
                    f"/-- {sample.informal_problem or ''}-/\n{formal_prefix}\nall_goals sorry")
        return self.prompt_template.format(composed)


class GoedelProverV2Agent(WholeProofGenerationCoTAgent):
    agent_name = "goedel_prover_v2"
    token_limit = 8192

    def build_prompt(self, sample: FormalProblem, formal_prefix: str) -> str:
        composed = (f"{LEAN4_DEFAULT_IMPORTS}\n\n{OPTION_HEADER}\n\n"
                    f"/-- {sample.informal_problem or ''}-/\n{formal_prefix}")
        return f"""Complete the following Lean 4 code:

```lean4
{composed.strip()}
all_goals sorry```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""


class KiminaProverAgent(WholeProofGenerationCoTAgent):
    agent_name = "kimina_prover"
    token_limit = 8096

    def build_prompt(self, sample: FormalProblem, formal_prefix: str) -> str:
        composed = (f"{LEAN4_DEFAULT_IMPORTS}\n\n{OPTION_HEADER}\n\n"
                    f"/-- {sample.informal_problem or ''}-/\n{formal_prefix.rstrip()}")
        prompt = "Think about and solve the following problem step by step in Lean 4."
        prompt += f"\n# Problem:{sample.informal_problem or ''}"
        prompt += f"\n# Formal statement:\n```lean4\n{composed}\n```\n"
        return prompt

    def _format_thought_prompt(self, sample: FormalProblem, formal_prefix: str) -> str:
        tokenizer = self._get_tokenizer()
        user_content = self.build_prompt(sample, formal_prefix)
        messages = [
            {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True) + "<think>\n"

    def _format_proof_prompt(self, sample: FormalProblem, formal_prefix: str, thought: str) -> str:
        tokenizer = self._get_tokenizer()
        user_content = self.build_prompt(sample, formal_prefix)
        composed = (f"{LEAN4_DEFAULT_IMPORTS}\n\n{OPTION_HEADER}\n\n"
                    f"/-- {sample.informal_problem or ''}-/\n{formal_prefix.rstrip()}")
        messages = [
            {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
            {"role": "user", "content": user_content},
        ]
        return (tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
                + f"<think>\n{thought}</think>\n```lean4\n{composed}\n")

    async def prove_async(self, sample: FormalProblem, server: Any, init_state: GoalState, tag: str = "", verbose: bool = False) -> ProofResult:
        start_time = time.time()
        formal_prefix = server.formal_problem_framework_prefix
        thought_prompt = self._format_thought_prompt(sample, formal_prefix)
        attempts = []
        final_state = None
        final_proof = None
        log_debug = logger.info if verbose else logger.debug
        tokenizer = self._get_tokenizer()

        for sample_i in range(self.num_samples_per_trial):
            attempt_start = time.time()
            try:
                thought_len = len(tokenizer.tokenize(thought_prompt))
                assert thought_len < self.token_limit, f"Current prompt length: {thought_len}"
                response = await self.gen_client.completions.create(
                    model=self.gen_model_name,
                    prompt=thought_prompt,
                    temperature=self.temperature, top_p=self.top_p, max_tokens=self.token_limit - thought_len,
                    stream=False,
                    n=1,
                    stop="</think>",
                )
                thought = response.choices[0].text
                final_prompt = self._format_proof_prompt(sample, formal_prefix, thought)
                proof_len = len(tokenizer.tokenize(final_prompt))
                assert proof_len < self.token_limit, f"Current prompt length: {proof_len}"
                response = await self.gen_client.completions.create(
                    model=self.gen_model_name,
                    prompt=final_prompt,
                    temperature=self.temperature, top_p=self.top_p, max_tokens=self.token_limit - proof_len,
                    stream=False,
                    n=1,
                )
                proof = "{\n" + self.post_process(response.choices[0].text) + "\n}"

                assert all(token not in proof for token in BANNED_TOKENS), \
                    f"Banned token detected in proof: {proof}"

                state = await server.server.goal_tactic_async(init_state, 0, proof)
                success = state.is_solved
                attempts.append(ProofAttempt(
                    success=success,
                    duration=time.time() - attempt_start,
                    proof=proof,
                    response_content=response.choices[0].text,
                    final_state=state,
                ))
                if success:
                    final_state = state
                    final_proof = proof
                    break
            except Exception as exc:
                error = traceback.format_exc()
                attempts.append(ProofAttempt(
                    success=False,
                    duration=time.time() - attempt_start,
                    error=error))
                log_debug(f"prove({tag}, {sample_i}/{self.num_samples_per_trial}) failed: {repr(exc)}")

        success = final_state is not None and final_state.is_solved
        return ProofResult(
            success=success,
            duration=time.time() - start_time,
            proof=[(0, final_proof)] if success else [],
            final_state=final_state,
            attempts=attempts,
        )
