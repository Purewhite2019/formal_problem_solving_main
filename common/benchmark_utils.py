import json
from pathlib import Path
from typing import Any, Dict, List, Union

from common.pantograph.dataclasses import FormalProblem


BENCHMARK_FILE_MAP = {
    "formal_math500": "formal_math500.jsonl",
    "minif2f_solving": "minif2f_solving.jsonl",
    "putnam_solving": "putnam_solving.jsonl",
}


def resolve_benchmark_path(benchmark_root: str, benchmark: str) -> Path:
    path = Path(benchmark)
    if path.exists():
        return path

    benchmark = benchmark.lower()
    root = Path(benchmark_root)
    if benchmark not in BENCHMARK_FILE_MAP:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    jsonl_candidate = root / BENCHMARK_FILE_MAP[benchmark]
    if jsonl_candidate.exists():
        return jsonl_candidate
    raise FileNotFoundError(f"Cannot find benchmark jsonl file: {jsonl_candidate}")


def _load_jsonl_benchmark(path: Path) -> List[FormalProblem]:
    with path.open("r", encoding="utf-8") as handle:
        return [FormalProblem.from_kwargs(**json.loads(line)) for line in handle if line.strip()]


def load_benchmark_samples(benchmark_root: str, benchmark: str) -> List[FormalProblem]:
    path = resolve_benchmark_path(benchmark_root, benchmark)
    if path.is_file() and path.suffix == ".jsonl":
        return _load_jsonl_benchmark(path)
    raise ValueError(f"Unsupported benchmark path: {path}")


def _field(sample: Union[FormalProblem, Dict[str, Any]], key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _stable(value: Any) -> str:
    if hasattr(value, "serialize"):
        value = value.serialize()
    elif isinstance(value, list):
        value = [item.serialize() if hasattr(item, "serialize") else item for item in value]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def sample_resume_key(sample: Union[FormalProblem, Dict[str, Any]]) -> tuple:
    if isinstance(sample, list):
        sample = sample[0] if sample else {}
    return (
        _field(sample, "informal_problem"),
        _field(sample, "informal_answer"),
        _field(sample, "informal_solution"),
        _stable(_field(sample, "independent_variables")),
        _stable(_field(sample, "hypotheses")),
        _stable(_field(sample, "conclusions")),
        _field(sample, "formal_answer"),
        _field(sample, "formal_answer_type"),
    )
