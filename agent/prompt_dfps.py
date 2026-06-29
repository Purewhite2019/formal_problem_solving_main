import time
import traceback
from typing import Any

from loguru import logger
from openai import AsyncOpenAI, NOT_GIVEN

from agent.base import ProofAttempt, ProofResult, ProverAgent
from common.pantograph.dataclasses import FormalProblem, GoalState, TacticDraft
from common.utils import remove_comments, strip_code


def dfps_forward_context(initial_solution_state: GoalState) -> str:
    goal = str(initial_solution_state.goals[0])
    if goal.startswith("case "):
        goal = "\n".join(goal.splitlines()[1:])
    return "\n".join(goal.splitlines()[:-1] + ["⊢ ?Answer"]) if not goal.endswith("⊢ ?Answer") else goal


def dfps_sample(informal_problem: str, forward_context: str, formal_solution: str | None = None, keep_comments: bool = True) -> str:
    text = f'''# Informal Problem
"""
{informal_problem.strip()}
"""
# Goal State
```lean4
{forward_context.strip()}
```
'''
    if formal_solution is not None:
        if not keep_comments:
            formal_solution = "\n".join(line for line in remove_comments(formal_solution).splitlines() if line.strip())
        text += f'''# Formal Proof
```lean4
{formal_solution.strip()}
```
'''
    return text


def dfps_prompt(prompt_style: str, demonstrations: list[dict], informal_problem: str, initial_solution_state: GoalState, no_tool_call: bool = True) -> str:
    prompt_style = prompt_style.lower()
    forward_context = dfps_forward_context(initial_solution_state)
    instruction = "please construct a formal proof deducing the answer"
    keep_comments = False
    if prompt_style == "hybrid_cot":
        instruction = "please think step by step and construct a formal proof deducing the answer"
        keep_comments = True

    if prompt_style not in {"direct_icl", "hybrid_cot"}:
        raise ValueError(f"Unsupported D-FPS prompt style: {prompt_style}")

    tool_line = "\nPlease DO NOT call any tool." if no_tool_call else ""
    prompt = f'''Given an informal math problem and a corresponding Lean 4 goal state, {instruction}.{tool_line}
Please assume the following header code has already been executed, and do not add any imports or openings.
```lean4
import Mathlib
import Aesop
```

Here are some examples:

'''
    for demonstration in demonstrations:
        prompt += dfps_sample(
            demonstration["informal_problem"],
            demonstration["forward_context"],
            demonstration["formal_solution"],
            keep_comments=keep_comments,
        )
        prompt += "\n---\n\n"
    prompt += "Now, please generate a formal proof for the following problem.\n\n"
    prompt += dfps_sample(informal_problem, forward_context)
    return prompt


class PromptDFPSAgent(ProverAgent):
    agent_name = "prompt_dfps"
    prompt_style = "direct_icl"

    def __init__(
        self,
        gen_client: AsyncOpenAI,
        gen_model_name: str,
        demonstrations: list[dict],
        try_num: int = 16,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        no_tool_call: bool = True,
        **kwargs,
    ) -> None:
        self.gen_client = gen_client
        self.gen_model_name = gen_model_name
        self.demonstrations = demonstrations
        self.try_num = try_num
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.no_tool_call = no_tool_call

    async def prove_async(
        self,
        sample: FormalProblem,
        server: Any,
        init_state: GoalState,
        tag: str = "",
        verbose: bool = False,
    ) -> ProofResult:
        start_time = time.time()
        attempts: list[ProofAttempt] = []
        log_debug = logger.info if verbose else logger.debug

        for try_i in range(self.try_num):
            attempt_start = time.time()
            try:
                prompt = dfps_prompt(self.prompt_style, self.demonstrations, sample.informal_problem or "", init_state, no_tool_call=self.no_tool_call)
                response = await self.gen_client.chat.completions.create(
                    model=self.gen_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens if self.max_tokens > 0 else NOT_GIVEN,
                    stream=False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    n=1,
                )
                content = response.choices[0].message.content or ""
                formal_solution = strip_code(content)
                solved_state = await server.server.goal_tactic_async(init_state, 0, TacticDraft("by {\n" + formal_solution + "\n}"))
                success = solved_state.is_solved
                attempts.append(ProofAttempt(
                    success=success,
                    duration=time.time() - attempt_start,
                    proof=formal_solution,
                    response_content=content,
                    final_state=solved_state,
                ))
                if success:
                    return ProofResult(
                        success=True,
                        duration=time.time() - start_time,
                        proof=formal_solution,
                        final_state=solved_state,
                        attempts=attempts,
                    )
            except Exception as exc:
                error = traceback.format_exc()
                attempts.append(ProofAttempt(success=False, duration=time.time() - attempt_start, error=error))
                log_debug(f"prove({tag}-{try_i}) failed: {repr(exc)}")

        return ProofResult(success=False, duration=time.time() - start_time, attempts=attempts)


class DirectICLDFPSAgent(PromptDFPSAgent):
    agent_name = "direct_icl"
    prompt_style = "direct_icl"


class HybridCoTDFPSAgent(PromptDFPSAgent):
    agent_name = "hybrid_cot"
    prompt_style = "hybrid_cot"
