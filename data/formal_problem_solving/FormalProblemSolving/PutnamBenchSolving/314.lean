import FormalProblemSolving.Basic
open Nat

/-
## Problem
For a positive integer $n$, let $f_n(x) = \cos(x) \cos(2x) \cos(3x) \cdots \cos(nx)$. Find the smallest $n$ such that $|f_n''(0)| > 2023$.

## Answer
18
-/

-- # Formal Answer
noncomputable abbrev Q314.gt := fun (answer : ℕ) => answer = (18 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q314
find (answer : ℕ) s.t.
  (f : ℕ → ℝ → ℝ)
  (hf : ∀ n > 0, f n = fun x : ℝ => ∏ i in Finset.Icc 1 n, Real.cos (i * x))
: (IsLeast {n | 0 < n ∧ |iteratedDeriv 2 (f n) 0| > 2023} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q314
find_all (answer : ℕ) iff
  (f : ℕ → ℝ → ℝ)
  (hf : ∀ n > 0, f n = fun x : ℝ => ∏ i in Finset.Icc 1 n, Real.cos (i * x))
: (IsLeast {n | 0 < n ∧ |iteratedDeriv 2 (f n) 0| > 2023} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
