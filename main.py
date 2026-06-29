import asyncio
import traceback
from typing import Optional

from fire import Fire
from loguru import logger
from openai import AsyncOpenAI

from agent.base import ProofResult
from agent.prompt_dfps import DirectICLDFPSAgent, HybridCoTDFPSAgent
from agent.prompt_tpfps import PromptTPFPSAgent
from agent.proof_search import DummyProofSearchAgent, LeanSTaRAgent, StepProverAgent
from agent.whole_proof_generation import (
    DeepSeekProverV15Agent,
    DeepSeekProverV2NonCoTAgent,
    GoedelProverAgent,
    STPAgent,
    TheoremLlamaAgent,
)
from agent.whole_proof_generation_cot import (
    DeepSeekProverV2CoTAgent,
    GoedelProverV2Agent,
    KiminaProverAgent,
)
from common.evaluation import (
    load_jsonl,
    load_samples_with_resume,
    make_log_prefix,
    run_concurrently,
    save_results,
    setup_run,
    trim_trailing_sorry,
)
from common.pantograph.dataclasses import FormalProblem
from common.pantograph.solving_server import PropSolvingServer, TermSolvingServer


AGENT_REGISTRY = {
    "dummy": DummyProofSearchAgent,
    "stepprover": StepProverAgent,
    "leanstar": LeanSTaRAgent,
    "deepseek_prover_v15": DeepSeekProverV15Agent,
    "deepseek_prover_v1_5": DeepSeekProverV15Agent,
    "deepseek_prover_v2_cot": DeepSeekProverV2CoTAgent,
    "deepseek_prover_v2": DeepSeekProverV2CoTAgent,
    "deepseek_prover_v2_noncot": DeepSeekProverV2NonCoTAgent,
    "goedel_prover": GoedelProverAgent,
    "goedel": GoedelProverAgent,
    "goedel_prover_v2": GoedelProverV2Agent,
    "kimina_prover": KiminaProverAgent,
    "kimina": KiminaProverAgent,
    "stp": STPAgent,
    "theoremllama": TheoremLlamaAgent,
    "prompt": PromptTPFPSAgent,
    "direct_icl": DirectICLDFPSAgent,
    "hybrid_cot": HybridCoTDFPSAgent,
}

PROOF_SEARCH_AGENTS = {"dummy", "stepprover", "leanstar"}
DFPS_AGENTS = {"direct_icl", "hybrid_cot"}
PROMPT_AGENTS = {"prompt"}


def main(
    log_root: str,
    task: str,
    agent: str,
    benchmark: str,
    benchmark_root: str = "data",
    project_root: str = "data/formal_problem_solving",
    gen_base_url: Optional[str] = None,
    gen_api_key: Optional[str] = None,
    gen_model_name: Optional[str] = None,
    solver_base_url: Optional[str] = None,
    solver_api_key: Optional[str] = None,
    solver_model_name: Optional[str] = None,
    demonstration_path: str = "data/prompt_hybrid_cot_final.jsonl",
    max_search_trials: int = 600,
    num_samples_per_trial: int = 128,
    num_tactics_per_trial: int = 32,
    try_num: int = 16,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    num_concurrency: int = 12,
    verbose: bool = False,
    dry_run: bool = False,
    resume_from: Optional[str] = None,
):
    saved_args = {**locals()}
    task = task.lower()
    agent = agent.lower()
    benchmark = benchmark.lower()
    if task not in {"tp", "fps", "d-fps"}:
        raise ValueError(f"Unsupported task: {task}")
    if agent not in AGENT_REGISTRY:
        raise ValueError(f"Unsupported agent: {agent}")
    if task == "d-fps" and agent not in DFPS_AGENTS:
        raise ValueError("task=d-fps requires agent=direct_icl or agent=hybrid_cot")
    if task != "d-fps" and agent in DFPS_AGENTS:
        raise ValueError("D-FPS agents require task=d-fps")

    model_name = solver_model_name or gen_model_name
    log_prefix = make_log_prefix(benchmark, task, AGENT_REGISTRY[agent].agent_name, model_name)
    now = setup_run(log_root, log_prefix, verbose, f"{benchmark} {task} {agent}", saved_args)
    log_debug = logger.info if verbose else logger.debug

    try:
        samples, finished = load_samples_with_resume(benchmark_root, benchmark, log_prefix, resume_from)
    except ValueError as exc:
        logger.error(str(exc))
        return

    if task == "d-fps":
        client = AsyncOpenAI(base_url=solver_base_url or gen_base_url, api_key=solver_api_key or gen_api_key)
        demonstrations = load_jsonl(demonstration_path)
        prover = AGENT_REGISTRY[agent](
            gen_client=client,
            gen_model_name=model_name or "",
            demonstrations=demonstrations,
            try_num=try_num,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    elif agent in PROOF_SEARCH_AGENTS:
        client = AsyncOpenAI(base_url=gen_base_url, api_key=gen_api_key)
        prover = AGENT_REGISTRY[agent](
            gen_client=client,
            gen_model_name=gen_model_name or "",
            max_search_trials=max_search_trials,
            num_samples_per_trial=num_tactics_per_trial,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        client = AsyncOpenAI(base_url=gen_base_url, api_key=gen_api_key)
        extra_kwargs = {}
        if agent in PROMPT_AGENTS:
            extra_kwargs["task_mode"] = task
        prover = AGENT_REGISTRY[agent](
            gen_client=client,
            gen_model_name=gen_model_name or "",
            num_samples_per_trial=num_samples_per_trial,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **extra_kwargs,
        )

    async def evaluate(sample: FormalProblem, tag_i: int) -> None:
        try:
            if task == "d-fps":
                server = PropSolvingServer(imports=["Mathlib", "Aesop"], project_path=project_root, timeout=300, tag=str(tag_i))
                init_state, _ = await server.init_forward_solving_state_async(sample)
            else:
                server = TermSolvingServer(imports=["Mathlib", "Aesop"], project_path=project_root, timeout=300, tag=str(tag_i))
                if task == "tp":
                    init_state = await server.init_backward_state_async(sample)
                else:
                    init_state, _ = await server.init_solving_state_async(sample)
                server.formal_problem_framework_prefix = trim_trailing_sorry(server.formal_problem_framework)

            logger.info(f"evaluate({tag_i}): initial state ```\n{init_state}\n```")
            if dry_run:
                finished.append({"sample": sample.serialize(), "result": None, "submission": None, "rpe_proof": None, "solved": (None if task == "tp" else False)})
                return

            result: ProofResult = await prover.prove_async(sample, server, init_state, tag=str(tag_i), verbose=verbose)
            submission = None
            rpe_proof = None
            proven = result.success
            solved = None if task == "tp" else False
            if task in {"fps", "d-fps"} and result.metadata and "solved" in result.metadata:
                solved = result.metadata["solved"]
                rpe_proof = result.metadata.get("rpe_proof")
            elif result.success and task in {"fps", "d-fps"} and result.final_state is not None:
                try:
                    submission = await server.get_submission_async(result.final_state)
                    rpe_proof = await server.prove_eq_async(submission)
                    solved = rpe_proof is not None
                except Exception:
                    logger.error(f"evaluate({tag_i}): RPE check failed:\n{traceback.format_exc()}")
                    solved = False

            finished.append({
                "sample": sample.serialize(),
                "result": result,
                "submission": submission,
                "rpe_proof": rpe_proof,
                "proven": proven,
                "solved": solved,
            })
            status = "solved" if solved else ("proven" if proven else "failed")
            logger.info(f"evaluate({tag_i}): {status} in {result.duration:.1f}s")
        except Exception:
            logger.error(f"evaluate({tag_i}) failed:\n{traceback.format_exc()}")

    try:
        asyncio.run(run_concurrently(samples, evaluate, num_concurrency, log_debug))
    finally:
        save_results(log_root, log_prefix, now, finished)


if __name__ == "__main__":
    Fire(main)
