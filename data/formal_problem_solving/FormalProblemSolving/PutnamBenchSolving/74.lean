import FormalProblemSolving.Basic
open Set Function Filter Topology Polynomial Real

/-
## Problem
Let $B_n(x) = 1^x + 2^x + \dots + n^x$ and let $f(n) = \frac{B_n(\log_n 2)}{(n \log_2 n)^2}$. Does $f(2) + f(3) + f(4) + \dots$ converge?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q74.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q74
find (answer : Prop) s.t.
  (B : ℕ → ℝ → ℝ)
  (hB : B = fun (n : ℕ) (x : ℝ) ↦ ∑ k in Finset.Icc 1 n, (k : ℝ) ^ x)
  (f : ℕ → ℝ)
  (hf : f = fun n ↦ B n (logb n 2) / (n * logb 2 n) ^ 2)
: (answer ↔ (∃ L : ℝ, Tendsto (fun N ↦ ∑ j in Finset.Icc 2 N, f j) atTop (𝓝 L)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q74
find_all (answer : Prop) iff
  (B : ℕ → ℝ → ℝ)
  (hB : B = fun (n : ℕ) (x : ℝ) ↦ ∑ k in Finset.Icc 1 n, (k : ℝ) ^ x)
  (f : ℕ → ℝ)
  (hf : f = fun n ↦ B n (logb n 2) / (n * logb 2 n) ^ 2)
: (answer ↔ (∃ L : ℝ, Tendsto (fun N ↦ ∑ j in Finset.Icc 2 N, f j) atTop (𝓝 L)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
