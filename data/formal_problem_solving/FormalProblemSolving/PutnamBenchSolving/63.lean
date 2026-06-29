import FormalProblemSolving.Basic
open Real

/-
## Problem
Let $C$ be the class of all real valued continuously differentiable functions $f$ on the interval $0 \leq x \leq 1$ with $f(0)=0$ and $f(1)=1$. Determine the largest real number $u$ such that $u \leq \int_0^1|f'(x)-f(x)|\,dx$ for all $f$ in $C$.

## Answer
$1/e$
-/

-- # Formal Answer
noncomputable abbrev Q63.gt := fun (answer : ℝ) => answer = ((1 : ℝ) / (rexp (1 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q63
find (answer : ℝ) s.t.
  (C : Set (ℝ → ℝ))
  (hC : C = {f : ℝ → ℝ | ContDiffOn ℝ 1 f (Set.Icc 0 1) ∧ f 0 = 0 ∧ f 1 = 1})
: (IsGreatest {u : ℝ | ∀ f ∈ C, u ≤ (∫ x in Set.Ioo 0 1, |deriv f x - f x|)} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q63
find_all (answer : ℝ) iff
  (C : Set (ℝ → ℝ))
  (hC : C = {f : ℝ → ℝ | ContDiffOn ℝ 1 f (Set.Icc 0 1) ∧ f 0 = 0 ∧ f 1 = 1})
: (IsGreatest {u : ℝ | ∀ f ∈ C, u ≤ (∫ x in Set.Ioo 0 1, |deriv f x - f x|)} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
