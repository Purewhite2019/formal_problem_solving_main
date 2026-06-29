---
license: apache-2.0
task_categories:
- text-generation
pretty_name: Formal Problem-Solving Benchmarks
size_categories:
- 1K<n<10K
---

# Dataset Card for Formal Problem-Solving Benchmarks

This dataset is part of the official implementation of _Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving_, accepted as an **ICML 2026 Spotlight** paper.

Links:

- Paper: https://openreview.net/forum?id=hgMZraPlSv
- Project: https://github.com/Purewhite2019/formal_problem_solving_main

## Overview

The benchmark supports three evaluation settings:

- **Formal Problem-Solving (FPS)**: Given a formal problem, generate a formal solution. The formal solution should solve all goals and provide a direct answer.
- **Deductive Formal Problem-Solving (D-FPS)**: Given a formal problem, generate a forward solution and, optionally, a backward proof. The forward solution should derive a direct answer and prove its completeness; the backward proof should prove the answer's soundness.
- **Formal Theorem Proving (FTP)**: Given a formal problem and its ground-truth answer, generate a formal proof of the ground-truth answer's correctness.

## Benchmarks

| JSONL File | Benchmark | Size |
| --- | --- | ---: |
| `formal_math500.jsonl` | FormalMath500 | 387 |
| `minif2f_solving.jsonl` | MiniF2F-Solving | 375 |
| `putnam_solving.jsonl` | PutnamBench-Solving | 324 |

**FormalMath500** is a formalized subset of MATH500 with 387 samples covering Algebra, Intermediate Algebra, Number Theory, Prealgebra, and Precalculus.

**MiniF2F-Solving** refactors MiniF2F propositions into problem-solving instances with detached direct answers. It contains 375 samples from AIME, MATH-Algebra, AMC, IMO, and MATH-Number Theory.

**PutnamBench-Solving** refactors a subset of PutnamBench into 324 undergraduate-level problem-solving instances across algebra, analysis, number theory, geometry, linear algebra, combinatorics, abstract algebra, probability, and set theory.

## Dataset Structure

Each JSONL record contains the following core fields:

- `informal_problem`: The natural-language problem statement, including LaTeX when applicable.
- `informal_answer`: The natural-language ground-truth answer, including LaTeX when applicable.
- `header`: Lean openings/options used before the formal problem.
- `independent_variables`: Problem-level independent variables.
- `hypotheses`: Assumptions rendered in the Lean `s.t.` / `iff` block before the conclusion.
- `conclusions`: Lean propositions that the queried answer must satisfy.
- `formal_answer`: The formal ground-truth answer predicate in Lean 4.
- `formal_answer_type`: The Lean type of the direct answer.
- `metainfo`: Benchmark name and sample index.

The released JSONL files intentionally omit empty draft fields and Lean-only export override fields. The structured JSONL files can be exported to the Lean benchmark project with `common/export_fps_benchmark.py` in the code repository.

## License

This dataset is released under the Apache 2.0 license.

## Citation

If you find this dataset useful, please cite:

```bibtex
@inproceedings{
liu2026beyond,
title={Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving},
author={Qi Liu and Xinhao Zheng and Renqiu Xia and Xingzhi Qi and Qinxiang Cao and Junchi Yan},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=hgMZraPlSv}
}
```
