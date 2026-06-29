# Formal Problem-Solving Lean Benchmark

Official Lean benchmark for the ICML 2026 Spotlight paper **"Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving"**.

OpenReview: https://openreview.net/forum?id=hgMZraPlSv

This Lean project contains **1086 formally specified mathematical problems** designed to evaluate formal problem-solving capabilities beyond traditional theorem proving. The problems can be used with both FPS and D-FPS formulations.

## Example

Please see [DFPS_Example.lean](FormalProblemSolving/DFPS_Example.lean) and [FPS_Example.lean](FormalProblemSolving/FPS_Example.lean) for example formalizations of problem-solving in D-FPS and FPS, respectively.

## Project Structure

```text
formal_problem_solving/
├── FormalProblemSolving/
│   ├── Basic.lean              # Core FPS/DFPS framework definitions
│   ├── DFPS_Example.lean       # Example D-FPS problem
│   ├── FPS_Example.lean        # Example FPS problem
│   ├── FormalMath500/          # 387 FormalMath500 problems
│   ├── MiniF2FSolving/         # 375 MiniF2F-Solving problems
│   └── PutnamBenchSolving/     # 324 PutnamBench-Solving problems
├── lakefile.toml               # Lake project configuration
└── README.md
```

## Source

The Lean files are exported from the structured JSONL benchmark files in the parent `data/` directory. To regenerate this project from the repository root, run:

```shell
python common/export_fps_benchmark.py \
  --data_root data \
  --output_root data/formal_problem_solving \
  --benchmark all
```

Then build the Lean project:

```shell
cd data/formal_problem_solving
lake build
```

## License

See [LICENSE](LICENSE) for details.
