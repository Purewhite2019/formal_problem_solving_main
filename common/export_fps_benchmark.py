import argparse
import json
from pathlib import Path


BENCHMARKS = {
    "formal_math500": ("FormalMath500", "formal_math500.jsonl"),
    "minif2f_solving": ("MiniF2FSolving", "minif2f_solving.jsonl"),
    "putnam_solving": ("PutnamBenchSolving", "putnam_solving.jsonl"),
}


LEAN_CODE_TEMPLATE = r'''import FormalProblemSolving.Basic
{header_block}
/-
## Problem
{informal_problem}

## Answer
{informal_answer}
-/

-- # Formal Answer
noncomputable abbrev Q{p_index}.gt := fun{formal_answer_args} (answer : {formal_answer_type}) => {formal_answer}

-- # Formal Problem (FPS)
namespace {benchmark_name}_FPS
open scoped FPS

problem Q{p_index} {V}
find (answer : {formal_answer_type}) s.t.{Phi}
: {Psi}
:= solve
-- Placeholder
all_goals sorry

end {benchmark_name}_FPS

-- # Formal Problem (D-FPS)
namespace {benchmark_name}_DFPS
open scoped DFPS

problem Q{p_index} {V}
find_all (answer : {formal_answer_type}) iff{Phi}
: {Psi}
:= solve
case Answer := sorry
all_goals sorry

end {benchmark_name}_DFPS
'''


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_problem(record: dict, benchmark_name: str, index: int) -> str:
    variables = [f"  ({var['name'] or '_'} : {var['t']})" for var in record["independent_variables"]]
    hypotheses = [f"  ({var['name'] or '_'} : {var['t']})" for var in record["hypotheses"]]
    header = (record.get("header") or "").strip()
    formal_answer_args = (" " if variables else "") + " ".join(var.strip() for var in variables)
    lean_code = LEAN_CODE_TEMPLATE.replace(r"{p_index}", str(index)).format(
        header_block=header + "\n" if header else "",
        informal_problem=(record.get("informal_problem") or "").strip(),
        informal_answer=str(record.get("informal_answer") or "").strip(),
        formal_answer_args=formal_answer_args,
        formal_answer=record["formal_answer"],
        formal_answer_type=record["formal_answer_type"],
        V=("\n" if variables else "") + "\n".join(variables),
        Phi=("\n" if hypotheses else "") + "\n".join(hypotheses),
        Psi=" ∧ ".join(record["conclusions"]).strip(),
        benchmark_name=benchmark_name,
    )
    return "\n".join(line.rstrip() for line in lean_code.splitlines()) + "\n"


def export_benchmark(data_root: Path, output_root: Path, benchmark: str) -> None:
    benchmark_name, filename = BENCHMARKS[benchmark]
    records = load_jsonl(data_root / filename)
    destination = output_root / "FormalProblemSolving" / benchmark_name
    destination.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, 1):
        (destination / f"{index}.lean").write_text(render_problem(record, benchmark_name, index), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FPS JSONL benchmarks to the Lean Formal Problem-Solving project.")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--output_root", type=Path, required=True, help="Path to the Lean benchmark project root.")
    parser.add_argument("--benchmark", choices=["all", *BENCHMARKS.keys()], default="all")
    args = parser.parse_args()

    benchmarks = BENCHMARKS.keys() if args.benchmark == "all" else [args.benchmark]
    for benchmark in benchmarks:
        export_benchmark(args.data_root, args.output_root, benchmark)


if __name__ == "__main__":
    main()
