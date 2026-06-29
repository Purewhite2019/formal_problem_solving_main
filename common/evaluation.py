import asyncio
import json
import os
import os.path as osp
import pickle
import sys
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from loguru import logger
from tqdm import tqdm

from common.benchmark_utils import load_benchmark_samples, resolve_benchmark_path, sample_resume_key


def llm_name_from_path(model_name: Optional[str]) -> str:
    if not model_name:
        return "unknown-llm"
    return osp.basename(model_name.rstrip("/")) or model_name


def make_log_prefix(benchmark: str, task: str, agent_name: str, model_name: Optional[str]) -> str:
    return f"{benchmark}.{task}.{agent_name}.{llm_name_from_path(model_name)}."


def setup_run(log_root: str, log_prefix: str, verbose: bool, description: str, saved_args: dict) -> str:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_root, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO" if not verbose else "DEBUG")
    logger.add(osp.join(log_root, log_prefix + now + ".log"), level="DEBUG")
    logger.info(f"Running {description} with hyperparams: {saved_args}")
    return now


def load_samples_with_resume(
    benchmark_root: str,
    benchmark: str,
    log_prefix: str,
    resume_from: Optional[str],
) -> tuple[list, list]:
    benchmark_path = resolve_benchmark_path(benchmark_root, benchmark)
    samples = load_benchmark_samples(benchmark_root, benchmark)
    finished = []
    logger.info(f"Loaded {len(samples)} samples from {benchmark_path}")

    if resume_from is None:
        return samples, finished

    load_files = [p for p in os.listdir(resume_from) if p.startswith(log_prefix) and p.endswith(".pkl")]
    if len(load_files) != 1:
        raise ValueError(f'Expected one resumable pkl in "{resume_from}", found {load_files}')
    with open(osp.join(resume_from, load_files[0]), "rb") as handle:
        finished = pickle.load(handle)
    finished_keys = {sample_resume_key(item["sample"] if isinstance(item, dict) and "sample" in item else item) for item in finished}
    samples = [sample for sample in samples if sample_resume_key(sample) not in finished_keys]
    logger.warning(f"Resumed {len(finished)} samples; {len(samples)} remaining.")
    return samples, finished


async def run_concurrently(
    samples: list,
    worker: Callable[[Any, int], Awaitable[None]],
    num_concurrency: int,
    log_debug: Callable[[str], None],
) -> None:
    pending: set[asyncio.Task] = set()
    for i, sample in tqdm(enumerate(samples), total=len(samples)):
        if len(pending) >= num_concurrency:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception() is not None:
                    log_debug(f"Task exception: {task.exception()}")
        pending.add(asyncio.create_task(worker(sample, i)))
    if pending:
        await asyncio.wait(pending)
    await logger.complete()


def serialize_for_json(value: Any) -> Any:
    if hasattr(value, "serialize"):
        return serialize_for_json(value.serialize())
    if isinstance(value, dict):
        return {k: serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [serialize_for_json(v) for v in value]
    return value


def save_results(log_root: str, log_prefix: str, now: str, finished: list) -> None:
    logger.info(f"Saving results to {osp.join(log_root, log_prefix + now + '.(pkl|jsonl)')}")
    with open(osp.join(log_root, log_prefix + now + ".pkl"), "wb") as handle:
        pickle.dump(finished, handle)
    with open(osp.join(log_root, log_prefix + now + ".jsonl"), "w", encoding="utf-8") as handle:
        for record in finished:
            handle.write(json.dumps(serialize_for_json(record), ensure_ascii=False) + "\n")

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trim_trailing_sorry(formal_problem_framework: str) -> str:
    suffix = "all_goals sorry"
    if formal_problem_framework.endswith(suffix):
        return formal_problem_framework[: -len(suffix)]
    return formal_problem_framework
