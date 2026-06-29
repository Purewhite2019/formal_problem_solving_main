import time
import traceback
from typing import Any, Optional

from loguru import logger
from openai import AsyncOpenAI, NOT_GIVEN
from transformers import AutoTokenizer

from agent.base import ProofAttempt, ProofResult, ProverAgent
from common.constants import BANNED_TOKENS, CORE_OPTIONS
from common.pantograph.dataclasses import FormalProblem, GoalState
from common.utils import clean_proof, remove_comments


LEAN4_DEFAULT_IMPORTS = "import Mathlib\nimport Aesop"
OPTION_HEADER = "\n".join("set_option " + option.replace("=", " ") for option in CORE_OPTIONS)


def remove_min_indent_until_fence(output: str) -> str:
    code = output[: output.find("```")] if "```" in output else output
    lines = code.splitlines()
    min_spaces = min((len(line) - len(line.lstrip(" ")) for line in lines if line.strip()), default=0)
    return "\n".join(line[min_spaces:] for line in lines)


def lean_code_with_context(sample: FormalProblem, formal_statement: str) -> str:
    informal_prefix = f"/-- {sample.informal_problem}-/\n" if sample.informal_problem else ""
    return f"{LEAN4_DEFAULT_IMPORTS}\n\n{OPTION_HEADER}\n\n{informal_prefix}{formal_statement.rstrip()}"


class WholeProofGenerationAgent(ProverAgent):
    agent_name = "whole_proof"

    def __init__(
        self,
        gen_client: AsyncOpenAI,
        gen_model_name: str,
        num_samples_per_trial: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        **kwargs,
    ) -> None:
        self.gen_client = gen_client
        self.gen_model_name = gen_model_name
        self.num_samples_per_trial = num_samples_per_trial
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def build_prompt(self, sample: FormalProblem, formal_prefix: str) -> str:
        raise NotImplementedError

    def post_process(self, output: str) -> str:
        return clean_proof(output)

    async def generate(self, prompt: str) -> str:
        response = await self.gen_client.chat.completions.create(
            model=self.gen_model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens if self.max_tokens > 0 else NOT_GIVEN,
            stream=False,
            temperature=self.temperature,
            top_p=self.top_p,
            n=1,
        )
        return response.choices[0].message.content or ""

    async def prove_async(
        self,
        sample: FormalProblem,
        server,
        init_state: GoalState,
        tag: str = "",
        verbose: bool = False,
    ) -> ProofResult:
        start_time = time.time()
        prompt = self.build_prompt(sample, server.formal_problem_framework_prefix)
        attempts: list[ProofAttempt] = []
        final_state = None
        final_proof: Optional[str] = None
        log_debug = logger.info if verbose else logger.debug

        for sample_i in range(self.num_samples_per_trial):
            attempt_start = time.time()
            try:
                output = await self.generate(prompt)
                proof = "{\n" + self.post_process(output) + "\n}"
                assert all(token not in proof for token in BANNED_TOKENS), f"Banned token detected in proof: {proof}"
                state = await server.server.goal_tactic_async(init_state, 0, proof)
                success = state.is_solved
                attempts.append(ProofAttempt(
                    success=success,
                    duration=time.time() - attempt_start,
                    proof=proof,
                    response_content=output,
                    final_state=state,
                ))
                if success:
                    final_state = state
                    final_proof = proof
                    break
            except Exception as exc:
                error = traceback.format_exc()
                attempts.append(ProofAttempt(success=False, duration=time.time() - attempt_start, error=error))
                log_debug(f"prove({tag}, {sample_i}/{self.num_samples_per_trial}) failed: {repr(exc)}")

        success = final_state is not None and final_state.is_solved
        return ProofResult(
            success=success,
            duration=time.time() - start_time,
            proof=[(0, final_proof)] if success else [],
            final_state=final_state,
            attempts=attempts,
        )


class CompletionWholeProofGenerationAgent(WholeProofGenerationAgent):
    stop = "```"

    async def generate(self, prompt: str) -> str:
        response = await self.gen_client.completions.create(
            model=self.gen_model_name,
            prompt=prompt,
            max_tokens=self.max_tokens if self.max_tokens > 0 else NOT_GIVEN,
            stream=False,
            temperature=self.temperature,
            top_p=self.top_p,
            n=1,
            stop=self.stop,
        )
        return response.choices[0].text


class DeepSeekProverV15Agent(CompletionWholeProofGenerationAgent):
    agent_name = "deepseek_prover_v15"

    def build_prompt(self, sample: FormalProblem, formal_prefix: str) -> str:
        return (
            "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n"
            f"```lean4\n{lean_code_with_context(sample, formal_prefix)}"
        )


class GoedelProverAgent(CompletionWholeProofGenerationAgent):
    agent_name = "goedel_prover"

    def build_prompt(self, sample: FormalProblem, formal_prefix: str) -> str:
        return (
            "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n"
            f"```lean4\n{lean_code_with_context(sample, formal_prefix)}"
        )


class STPAgent(CompletionWholeProofGenerationAgent):
    agent_name = "stp"

    def build_prompt(self, sample: FormalProblem, formal_prefix: str) -> str:
        header = (sample.header or "").strip()
        return f"Complete the following Lean 4 code:\n\n```lean4\n{header}\n\n{formal_prefix.rstrip()}".strip()


class TheoremLlamaAgent(CompletionWholeProofGenerationAgent):
    agent_name = "theoremllama"

    def build_prompt(self, sample: FormalProblem, formal_prefix: str) -> str:
        formal_statement = formal_prefix.rstrip()
        return f"""Natural language version of theorem and proof:
{sample.informal_problem}
Show that it is {sample.informal_answer}.

{sample.informal_solution or ""}

### Lean4 version of theorem statement:
```lean
{formal_statement}
```

### Lean4 version of theorem and proof:
```lean
{formal_statement}"""


class DeepSeekProverV2NonCoTAgent(WholeProofGenerationAgent):
    agent_name = "deepseek_prover_v2_noncot"

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

    def build_prompt(self, sample: FormalProblem, formal_prefix: str) -> str:
        formal_statement = formal_prefix.rstrip()
        composed = (f"{LEAN4_DEFAULT_IMPORTS}\n\n{OPTION_HEADER}\n\n"
                    f"/-- {sample.informal_problem or ''}-/\n{formal_statement}\nall_goals sorry")
        return f"""Complete the following Lean 4 code:

```lean4
{composed}
```""".strip()

    def _format_prompt(self, sample: FormalProblem, formal_prefix: str) -> str:
        tokenizer = self._get_tokenizer()
        formal_statement = formal_prefix.rstrip()
        user_content = self.build_prompt(sample, formal_prefix)
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + f'''
```lean4
{formal_statement}
'''.strip()

    async def prove_async(self, sample: FormalProblem, server: Any, init_state: GoalState, tag: str = "", verbose: bool = False) -> ProofResult:
        start_time = time.time()
        formal_prefix = server.formal_problem_framework_prefix
        prompt = self._format_prompt(sample, formal_prefix)
        attempts = []
        final_state = None
        final_proof = None
        log_debug = logger.info if verbose else logger.debug

        for sample_i in range(self.num_samples_per_trial):
            attempt_start = time.time()
            try:
                response = await self.gen_client.completions.create(
                    model=self.gen_model_name,
                    prompt=prompt,
                    max_tokens=self.max_tokens if self.max_tokens > 0 else NOT_GIVEN,
                    stream=False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    n=1,
                    stop="```",
                )
                output = response.choices[0].text
                proof = "{\n" + remove_comments(remove_min_indent_until_fence(output)) + "\n}"
                assert all(token not in proof for token in BANNED_TOKENS), \
                    f"Banned token detected in proof: {proof}"

                state = await server.server.goal_tactic_async(init_state, 0, proof)
                success = state.is_solved
                attempts.append(ProofAttempt(
                    success=success,
                    duration=time.time() - attempt_start,
                    proof=proof,
                    response_content=output,
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
