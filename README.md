<div align="center">
    <h1> <a href="https://arxiv.org/abs/2505.04528">Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving</a></h1>

  <p align="center" style="font-size: 30px">
    <a href="https://arxiv.org/abs/2505.04528">📃Paper</a> • 
    <a href="https://huggingface.co/collections/purewhite42/formal-problem-solving-681b573aac8f09f308bb7c66">🤗Data</a> • 
    <a href="#-citation">📖Citation
  </p>
  <br>
  <img width="95%" src=assets/informal-formal.png>
</div>

## 🚩 News

- [2024/5/8] Release benchmark collection. See [here](https://huggingface.co/collections/purewhite42/formal-problem-solving-681b573aac8f09f308bb7c66).
- [2025/5/8] Upload [paper](https://arxiv.org/abs/2505.04528) and init [project](https://github.com/Purewhite2019/formal_problem_solving_main). 

## 🏃 Intro FPS

Our research focuses on:
1. What is problem-solving?
2. Beyond proving known targets, how can process-verified problem-solving be conducted inside existing formal theorem proving (FTP) environments?

## 💡 Contribution
- A principled formulation of problem-solving as a deterministic Markov decision process;
- **FPS** (_**F**ormal **P**roblem-**S**olving_), utilizing FTP (formal theorem proving) environments to perform process-verified problem-solving;
- **D-FPS** (_**D**eductive **FPS**_), decoupling solving and answer verification for better human-alignment;
- **RPE** (_**R**estricted **P**ropositional **E**quivalence_), a symbolic approach to determine the _correctness_ of answers by formal verification;
- Three benchmarks on problem-solving: **FormalMath500**, **MiniF2F-Solving** and **PutnamBench-Solving**.

## ⚡ Requirements
- [1] [Lean 4](https://github.com/leanprover/lean4): `v4.15.0`
- [2] [Mathlib 4](https://github.com/leanprover-community/mathlib4): `v4.15.0`
- [3] [Aesop](https://github.com/leanprover-community/aesop): `v4.15.0`
- [4] [Pantograph](https://github.com/lenianiva/Pantograph): `v0.2.25`

Please install Pantograph and link `/path/to/Pantograph/.lake/build/bin/repl` to `common/pantograph/pantograph-repl`

## 📁 Benchmarks
### Details
- **FormalMath500** is a formalized subset of the prevalent MATH500 benchmark[5,6], including 387 data points:
    - 123 about `Algebra`
    - 92 about `Intermediate Algebra`
    - 62 about `Number Theory`
    - 65 about `Prealgebra`
    - 45 about `Precalculus`

- **MiniF2F-Solving** is a refactored subset of MiniF2F[7], containing in 375 data points with:
    - 30 from `AIME`
    - 140 from `MATH-Algebra`
    - 82 from `AMC`
    - 3 from `IMO`
    - 120 from `MATH-Number Theory`

- **PutnamBench-Solving** is a refactored subset of PutnamBench[8], containing 324 data points with:
    - 9 about `Abstract Algebra`
    - 138 about `Algebra`
    - 122 about `Analysis`
    - 14 about `Combinatorics`
    - 28 about `Geometry`
    - 25 about `Linear Algebra`
    - 49 about `Number Theory`
    - 8 about `Probability`
    - 4 about `Set Theory`

### Direct Use
- **Formal Problem-Solving (FPS)**: Given a formal problem, generate a formal solution. The formal solution should solve all goals and provide a direct answer.

- **Deductive Formal Problem-Solving (D-FPS)**: Given a formal problem, generate a forward solution and, optionally, a backward proof. The forward solution should use deductive reasoning to derive a direct answer and prove its completeness.
The backward proof should prove the answer's soundness.

- **Formal Theorem Proving (FTP)**: Given a formal problem and its ground-truth answer, generate a formal proof to prove the ground-truth's correctness.

### Structures
Each problem contains the following fields:
- `informal_problem`: The problem in natural language (including LaTeX).
- `informal_answer`: The ground-truth answer in natural language (including LaTeX).
- `informal_solution`: A step-by-step solution in natural language (including LaTeX). 
- `header`: Code that should be executed before initializing the formal problem, e.g., `open`s. If `null`, `open BigOperators Real Nat Topology` should be used.
- `intros`: Independent variables $V$ and hypotheses $\Phi$. $V=\\{v_i\\}\_{i=1}^n$ is the set of variables independent to the queriable $a$. $\Phi = \\{\phi_i\\}\_{i=1}^p$ is the set of propositions that depend on $V$ (whose all free variables are included in $V$), consisting of conditions that can be used to deduce the answer.
- `outros`: Conclusions $\Psi = \\{\psi_i\\}\_{i=1}^q$ is the set of propositions which depend on $V \cup \\{a\\}$, consisting of conclusions that should be satisfied.
- `formal_answer`: The ground-truth answer in formal language (Lean 4).
- `formal_answer_type`: The type of the ground-truth answer in formal language (Lean 4).
- `metainfo`: Meta-information of the problem.

## 💻 Evaluation

Please run the following commands to reproduce baseline experiments.

\*For Proof Search and Whole-Proof Generation, set `only_proving=False` to evaluate FPS (finding a correct answer + proving its correctness) and `only_proving=True` to evaluate FTP (proving the correctness of the ground-truth answer).

### Proof Search (FPS)
- `model=stepprover_vanilla_avggoal`: Best-first Search w/ InternLM2.5-StepProver
- `model=leanstar_vanilla_avggoal`: Best-first Search w/ LeanSTaR

```shell
ulimit -s unlimited
for FPS_BENCHMARK in formal_math500 minif2f_solving putnam_solving;
do
    python -m evaluator.fps_proof_search \
            --log_root /path/to/output \
            --model "model_to_evaluate" \
            --benchmark ${FPS_BENCHMARK} \
            --project_root /path/to/mathlib4 \
            --max_search_trials 600 \
            --temperature 0.7 \
            --max_tokens 256 \
            --num_concurrency 32 \
            --verbose False \
            --gen_base_url "https://url.to.openai-style.api.server/" \
            --gen_api_key "gen_api_key" \
            --gen_model_name "name/of/proof-search-model" \
            --only_proving False
done
```

### Whole-Proof Generation (FPS)
- `model=deepseek_prover`: DeepSeekProver-V1.5
- `model=theoremllama`: TheoremLlama

```shell
ulimit -s unlimited
for FPS_BENCHMARK in formal_math500 minif2f_solving putnam_solving;
do
    python -m evaluator.fps_whole_proof_generation \
            --log_root /path/to/output \
            --model "model_to_evaluate" \
            --benchmark ${FPS_BENCHMARK} \
            --project_root /path/to/mathlib4 \
            --num_samples_per_trial 128 \
            --temperature 1.0 \
            --top_p 0.95 \
            --max_tokens 2048 \
            --num_concurrency 8 \
            --verbose False \
            --gen_base_url "https://url.to.openai-style.api.server/" \
            --gen_api_key "gen_api_key" \
            --gen_model_name "name/of/whole-proof-generation-model" \
            --only_proving False
done
```

### Prompting Methods (D-FPS)
- In-Context Learning

```shell
ulimit -s unlimited
for FPS_BENCHMARK in formal_math500 minif2f_solving putnam_solving;
do
    ulimit -s unlimited
    python -m evaluator.fps_icl \
            --log_root /path/to/output \
            --benchmark ${FPS_BENCHMARK} \
            --project_root /path/to/mathlib4 \
            --num_samples_per_trial 16 \
            --temperature 1.0 \
            --top_p 0.95 \
            --max_tokens 8192 \
            --num_concurrency 16 \
            --verbose False \
            --gen_base_url "https://url.to.openai-style.api.server/" \
            --gen_api_key "api-key" \
            --gen_model_name "name/of/general-LLM"
done
```

- Hybrid CoT

```shell
ulimit -s unlimited
for FPS_BENCHMARK in formal_math500 minif2f_solving putnam_solving;
do
    ulimit -s unlimited
    python -m evaluator.fps_hybrid_cot \
            --log_root /path/to/output \
            --benchmark ${FPS_BENCHMARK} \
            --num_samples_per_trial 16 \
            --temperature 1.0 \
            --top_p 0.95 \
            --max_tokens 8192 \
            --num_concurrency 16 \
            --verbose False \
            --gen_base_url "https://url.to.openai-style.api.server/" \
            --gen_api_key "api-key" \
            --gen_model_name "name/of/general-LLM"
done
```

## 📊 Baseline Performance
- _Solved_: indicates the portion that is successfully solved;
- _Proven_: indicates the portion whose statements (asserting the correctness of ground-truth answer) are proven;
- _NE-Submitted_: indicates the portion of problems whose submitted answers are incorrect under RPE (lower is better).

| Framework | Benchmark       | Method                 | Model                   | Solved↑ | Proven↑ | NE-Submitted↓ |
|-----------|-----------------|------------------------|-------------------------|---------|---------|---------------|
| FPS       | FormalMath500   | Proof Search           | InternLM2.5-StepProver  | 23.77%  | 47.55%  | 19.38%        |
|           |                 |                        | LeanSTaR                | 23.51%  | 43.41%  | 20.93%        |
|           |                 | Whole-Proof Generation | DeepSeekProver-V1.5     | 22.22%  | 46.51%  | 14.47%        |
|           |                 |                        | TheoremLlama            | 16.02%  | 4.39%   | 15.50%        |
|           | MiniF2F-Solving | Proof Search           | InternLM2.5-StepProver  | 27.47%  | 50.67%  | 13.60%        |
|           |                 |                        | LeanSTaR                | 24.27%  | 49.33%  | 14.40%        |
|           |                 | Whole-Proof Generation | DeepSeekProver-V1.5     | 22.40%  | 53.60%  | 10.93%        |
|           |                 |                        | TheoremLlama            | 13.07%  | 7.73%   | 8.80%         |
|           | Putnam-Solving  | Proof Search           | InternLM2.5-StepProver  | 0.00%   | 1.54%   | 28.09%        |
|           |                 |                        | LeanSTaR                | 0.00%   | 0.93%   | 41.05%        |
|           |                 | Whole-Proof Generation | DeepSeekProver-V1.5     | 0.31%   | 1.54%   | 22.22%        |
|           |                 |                        | TheoremLlama            | 0.00%   | 0.31%   | 16.67%        |
| D-FPS     | FormalMath500   | ICL                    | DeepSeek-V3             | 13.70%  |         | 0.00%         |
|           |                 | Hybrid CoT             | DeepSeek-V3             | 15.50%  |         | 1.03%         |
|           | MiniF2F-Solving | ICL                    | DeepSeek-V3             | 21.87%  |         | 0.00%         |
|           |                 | Hybrid CoT             | DeepSeek-V3             | 21.60%  |         | 0.00%         |
|           | Putnam-Solving  | ICL                    | DeepSeek-V3             | 0.00%   |         | 0.00%         |
|           |                 | Hybrid CoT             | DeepSeek-V3             | 0.00%   |         | 0.31%         |


## 📧 Contributing
This project is released under the Apache 2.0 license. Please see the [LICENSE](./LICENSE) file for more information.

Feel free to discuss the paper/data/code with us through issues/emails!
- Qi Liu: purewhite@sjtu.edu.cn

## 📖 Citation

If you find our work helps, please consider starring ⭐ us and citing:

```{bibtex}
@misc{liu2025theoremprovingformulationframework,
      title={Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving}, 
      author={Qi Liu and Xinhao Zheng and Renqiu Xia and Xingzhi Qi and Qinxiang Cao and Junchi Yan},
      year={2025},
      eprint={2505.04528},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.04528}, 
}
```

## References
[1] Moura, Leonardo de, and Sebastian Ullrich. "The Lean 4 theorem prover and programming language." Automated Deduction–CADE 28: 28th International Conference on Automated Deduction, Virtual Event, July 12–15, 2021, Proceedings 28. Springer International Publishing, 2021.

[2] Community, Mathlib . "The Lean mathematical library.", 10.1145/3372885.3373824. 2019.

[3] Limperg, Jannis, and Asta Halkjær From. "Aesop: White-box best-first proof search for Lean." Proceedings of the 12th ACM SIGPLAN International Conference on Certified Programs and Proofs. 2023.

[4] Aniva, Leni, et al. "Pantograph: A Machine-to-Machine Interaction Interface for Advanced Theorem Proving, High Level Reasoning, and Data Extraction in Lean 4." arXiv preprint arXiv:2410.16429 (2024).

[5] Lightman, Hunter, et al. "Let's verify step by step." The Twelfth International Conference on Learning Representations. 2023.

[6] Hendrycks, Dan, et al. "Measuring mathematical problem solving with the math dataset." arXiv preprint arXiv:2103.03874 (2021).

[7] Zheng, Kunhao, Jesse Michael Han, and Stanislas Polu. "Minif2f: a cross-system benchmark for formal olympiad-level mathematics." arXiv preprint arXiv:2109.00110 (2021).

[8] Tsoukalas, George, et al. "Putnambench: Evaluating neural theorem-provers on the putnam mathematical competition." arXiv preprint arXiv:2407.11214 (2024).
