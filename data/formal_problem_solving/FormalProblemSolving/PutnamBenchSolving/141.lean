import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
Find the maximum value of $\int_0^y \sqrt{x^4+(y-y^2)^2}\,dx$ for $0 \leq y \leq 1$.

## Answer
$\frac{1}{3}$
-/

-- # Formal Answer
noncomputable abbrev Q141.gt := fun (answer : ℝ) => answer = (1 / 3 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q141
find (answer : ℝ) s.t.
  (f : Set.Icc (0 : ℝ) 1 → ℝ)
  (hf : ∀ y : Set.Icc 0 1, f y = ∫ x in Set.Ioo 0 y, Real.sqrt (x ^ 4 + (y - y ^ 2) ^ 2))
: (IsGreatest (f '' (Set.Icc 0 1)) answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q141
find_all (answer : ℝ) iff
  (f : Set.Icc (0 : ℝ) 1 → ℝ)
  (hf : ∀ y : Set.Icc 0 1, f y = ∫ x in Set.Ioo 0 y, Real.sqrt (x ^ 4 + (y - y ^ 2) ^ 2))
: (IsGreatest (f '' (Set.Icc 0 1)) answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
