import FormalProblemSolving.Basic
open MeasureTheory

/-
## Problem
Let $R$ be the region consisting of the points $(x,y)$ of the cartesian plane satisfying both $|x|-|y| \leq 1$ and $|y| \leq 1$. Find the area of $R$.

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q116.gt := fun (answer : ℝ) => answer = (6 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q116
find (answer : ℝ) s.t.
  (R : Set (Fin 2 → ℝ))
  (hR : R = {p | |p 0| - |p 1| ≤ 1 ∧ |p 1| ≤ 1})
: ((volume R).toReal = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q116
find_all (answer : ℝ) iff
  (R : Set (Fin 2 → ℝ))
  (hR : R = {p | |p 0| - |p 1| ≤ 1 ∧ |p 1| ≤ 1})
: ((volume R).toReal = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
