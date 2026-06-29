import time
import traceback
from typing import Any, Optional
import regex as re

from loguru import logger
from openai import AsyncOpenAI, NOT_GIVEN

from agent.base import ProofAttempt, ProofResult
from agent.whole_proof_generation import WholeProofGenerationAgent
from common.constants import BANNED_TOKENS, CORE_OPTIONS, CODEBLOCK_PATTERN, H_SUBMISSION_NAME
from common.pantograph.dataclasses import FormalProblem, GoalState
from common.pantograph.server import Server
from common.utils import remove_comments, normalize_spaces


def _build_code_variables(sample, task_mode: str) -> tuple:
    V = [f"  ({v.name or '_'} : {v.t})" for v in sample.independent_variables]
    Phi = [f"  ({v.name or '_'} : {v.t})" for v in sample.hypotheses]
    Psi = sample.conclusions
    return V, Phi, Psi


TP_CODE_TEMPLATE = r'''{header}

/-
## Problem
{informal_problem}
-/

example {V}
(answer : {formal_answer_type}) ({H_SUBMISSION_NAME}: {formal_answer}){Phi}
: {Psi}
:= by
{proof}
'''

FPS_CODE_TEMPLATE = r'''open scoped FPS
{header}

/-
## Problem
{informal_problem}
-/

problem Q{p_index} {V}
find (answer : {formal_answer_type}) s.t.{Phi}
: {Psi}
:= solve
{proof}
'''

RPE_CODE_TEMPLATE = r'''
noncomputable def extracted_answer := extract_answer Q{p_index}
#print extracted_answer

noncomputable abbrev Q{p_index}.gt := fun{formal_answer_args} (answer : {formal_answer_type}) => {formal_answer}

lemma Q{p_index}.is_correct : (fun (answer : {formal_answer_type}) => answer = extracted_answer) = Q{p_index}.gt := by
  unfold Q{p_index}.gt
'''

RPE_PROOF = r'''  try ext
  try funext
  repeat {
    try norm_num;
    try ring_nf;
    try rfl;
    try aesop;
    try rw_search;
  }'''

LLM_TP_PROMPT_TEMPLATE = r'''Given an informal math problem and a corresponding Lean 4 formal statement, please construct a formal proof.
Please assume the following header code has already been executed, and do not add any imports or openings.
```lean4
import Mathlib
import Aesop
```

Here is one example:

```lean4
/-
## Problem
Given three real numbers $a, b, c$ with $a > 0$, find the minimizer $x_0$ of $ax^2 + bx + c$ over $\mathbb R$.
-/

example (a b c : ℝ)
(x₀ : ℝ) ({H_SUBMISSION_NAME}: x₀ = (- b / (2 * a)))
  (ha : a > 0)
  (f : ℝ → ℝ)
  (hf : f = fun x => a * x^2 + b * x + c)
: (∀ x, f x ≥ f x₀)
:= by
```

# Formal Proof
```lean4
intro x
have h_factorization : (a * x^2 + b * x + c) - (a * x₀^2 + b * x₀ + c) = a * (x - x₀)^2 := by
    have ha_ne_zero : 2 * a ≠ 0 := by linarith
    simp [{H_SUBMISSION_NAME}]
    field_simp [ha_ne_zero]
    ring
rw [ge_iff_le, ← sub_nonneg]; subst hf
change 0 ≤ (a * x^2 + b * x + c) - (a * x₀^2 + b * x₀ + c)
rw [h_factorization]; apply mul_nonneg
· linarith
· apply sq_nonneg
```

Now, please generate a formal proof for the following problem.
Please DO NOT call any tool.
Please answer instantly and only output the formal proof (do not output `example ... := by`).

```lean4
{tp_code}
```
'''

LLM_FPS_PROMPT_TEMPLATE = r'''Given an informal math problem and a corresponding Lean 4 formal statement, please construct a formal proof.
Please assume the following header code has already been executed, and do not add any imports or openings.
The `problem ... find ... s.t. ... : ... := solve` is a syntactic sugar that creates two goals: `Answer` and `Proof`, representing the answer to the problem and a proof of the answer's correctness.
```lean4
import Mathlib
import Aesop
import FormalProblemSolving.Basic
```

Here is one example:

```lean4
/-
## Problem
Given three real numbers $a, b, c$ with $a > 0$, find the minimizer $x_0$ of $ax^2 + bx + c$ over $\mathbb R$.
-/

problem Q_quad_min (a b c : ℝ)
find (x₀ : ℝ) s.t.
  (ha : a > 0)
  (f : ℝ → ℝ)
  (hf : f = fun x => a * x^2 + b * x + c)
: (∀ x, f x ≥ f x₀) := solve
```

# Formal Proof
```lean4
intro x
let vertex := - b / (2 * a)
have h_factorization : (a * x^2 + b * x + c) - (a * vertex^2 + b * vertex + c) = a * (x - vertex)^2 := by
    have ha_ne_zero : 2 * a ≠ 0 := by linarith
    dsimp [vertex]
    field_simp [ha_ne_zero]
    ring
rw [ge_iff_le, ← sub_nonneg]; subst hf
case Answer := - b / (2 * a)
change 0 ≤ (a * x^2 + b * x + c) - (a * vertex^2 + b * vertex + c)
rw [h_factorization]; apply mul_nonneg
· linarith
· apply sq_nonneg
```

Now, please generate a formal proof for the following problem.
Please DO NOT call any tool.
Please answer instantly and only output the formal proof (do not output `problem ... find ... s.t. ... : ... := solve`).

```lean4
{fps_code}
```
'''


def _post_process(content: str) -> str:
    if len(re.findall("```", content)) == 1:
        content = content[:content.find("```")].rstrip()
    else:
        parse_result = re.findall(CODEBLOCK_PATTERN, content)
        if len(parse_result) > 0:
            content = parse_result[-1].rstrip()
    if ":= solve" in content:
        content = content.split(":= solve")[-1].rstrip()
    return content


class PromptTPFPSAgent(WholeProofGenerationAgent):
    agent_name = "prompt"

    def __init__(self, gen_client, gen_model_name, num_samples_per_trial=16,
                 temperature=1.0, top_p=0.95, max_tokens=2048, task_mode="fps", **kwargs):
        self.gen_client = gen_client
        self.gen_model_name = gen_model_name
        self.num_samples_per_trial = num_samples_per_trial
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.task_mode = task_mode

    def _build_code_and_prompt(self, sample: FormalProblem, tag: str = "") -> tuple[str, str, Optional[str]]:
        V, Phi, Psi = _build_code_variables(sample, self.task_mode)
        if self.task_mode == "tp":
            code = TP_CODE_TEMPLATE.format(
                header=(sample.header or "").strip(),
                informal_problem=(sample.informal_problem or "").strip(),
                formal_answer_type=sample.formal_answer_type or "",
                H_SUBMISSION_NAME=H_SUBMISSION_NAME,
                formal_answer=sample.formal_answer or "",
                V=("\n" if V else "") + "\n".join(V),
                Phi=("\n" if Phi else "") + "\n".join(Phi),
                Psi=" ∧ ".join(Psi),
                proof="",
            ).strip()
            prompt = LLM_TP_PROMPT_TEMPLATE.format(
                H_SUBMISSION_NAME=H_SUBMISSION_NAME,
                tp_code=code,
            ).strip()
            return code, prompt, None

        code = FPS_CODE_TEMPLATE.replace(r"{p_index}", tag).format(
            header=(sample.header or "").strip(),
            informal_problem=(sample.informal_problem or "").strip(),
            formal_answer_type=sample.formal_answer_type or "",
            V=("\n" if V else "") + "\n".join(V),
            Phi=("\n" if Phi else "") + "\n".join(Phi),
            Psi=" ∧ ".join(Psi),
            proof="",
        ).strip()
        prompt = LLM_FPS_PROMPT_TEMPLATE.format(fps_code=code).strip()
        formal_answer_args = (" " if V else "") + " ".join(v.strip() for v in V)
        rpe_code = RPE_CODE_TEMPLATE.replace(r"{p_index}", tag).format(
            formal_answer_args=formal_answer_args,
            formal_answer=sample.formal_answer or "",
            formal_answer_type=sample.formal_answer_type or "",
        ).strip() + "\n" + RPE_PROOF
        return code, prompt, rpe_code

    def build_prompt(self, sample: FormalProblem, formal_prefix: str, tag: str = "") -> str:
        _, prompt, _ = self._build_code_and_prompt(sample, tag)
        return prompt

    async def prove_async(self, sample: FormalProblem, server: Any, init_state: GoalState, tag: str = "", verbose: bool = False) -> ProofResult:
        start_time = time.time()
        attempts = []
        log_debug = logger.info if verbose else logger.debug

        project_path = getattr(server, "project_path", None)
        server_timeout = getattr(server, "timeout", 300)
        code, prompt, rpe_code = self._build_code_and_prompt(sample, tag)

        for sample_i in range(self.num_samples_per_trial):
            attempt_start = time.time()
            try:
                response = await self.gen_client.chat.completions.create(
                    model=self.gen_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens if self.max_tokens > 0 else NOT_GIVEN,
                    stream=False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    n=1,
                )
                proof_text = _post_process(response.choices[0].message.content)
                assert all(token not in remove_comments(proof_text) for token in BANNED_TOKENS), \
                    f"Banned token detected in proof: {proof_text}"

                verify_server = await Server.create(
                    imports=["Mathlib", "Aesop", "FormalProblemSolving.Basic"],
                    project_path=project_path,
                    core_options=CORE_OPTIONS,
                    timeout=server_timeout,
                )
                proven = False
                metadata = {}
                try:
                    full_code = code + "\n" + proof_text
                    units = await verify_server.load_sorry_async(full_code)
                    proven = all(u.goal_state is None or u.goal_state.is_solved for u in units)
                    for u in units:
                        assert "error" not in str(getattr(u, "messages", [])), \
                            f"Error: {getattr(u, 'messages', [])}"

                    if proven and self.task_mode == "fps":
                        assert rpe_code is not None
                        full_code_with_rpe = code + "\n" + proof_text + "\n\n" + rpe_code
                        rpe_units = await verify_server.load_sorry_async(full_code_with_rpe)
                        rpe_solved = all(u.goal_state is None or u.goal_state.is_solved for u in rpe_units)
                        for u in rpe_units:
                            assert "error" not in str(getattr(u, "messages", [])), \
                                f"Error: {getattr(u, 'messages', [])}"
                        metadata = {"solved": rpe_solved, "rpe_proof": RPE_PROOF}
                finally:
                    verify_server._close()

                attempts.append(ProofAttempt(
                    success=proven,
                    duration=time.time() - attempt_start,
                    proof=proof_text,
                    response_content=response.choices[0].message.content,
                    final_state=None,
                ))
                if proven:
                    return ProofResult(
                        success=True,
                        duration=time.time() - start_time,
                        proof=proof_text,
                        final_state=None,
                        attempts=attempts,
                        metadata=metadata,
                    )
            except Exception as exc:
                error = traceback.format_exc()
                attempts.append(ProofAttempt(
                    success=False,
                    duration=time.time() - attempt_start,
                    error=error))
                log_debug(f"prove({tag}-{sample_i}) failed: {repr(exc)}")

        return ProofResult(success=False, duration=time.time() - start_time, attempts=attempts)
