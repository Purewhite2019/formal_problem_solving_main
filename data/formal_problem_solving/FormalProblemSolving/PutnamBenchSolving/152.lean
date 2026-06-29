import FormalProblemSolving.Basic

/-
## Problem
The horizontal line $y=c$ intersects the curve $y=2x-3x^3$ in the first quadrant as in the figure. Find $c$ so that the areas of the two shaded regions are equal. [Figure not included. The first region is bounded by the $y$-axis, the line $y=c$ and the curve; the other lies under the curve and above the line $y=c$ between their two points of intersection.]

## Answer
4/9
-/

-- # Formal Answer
noncomputable abbrev Q152.gt := fun (answer : ℝ) => answer = (4 / 9 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q152
find (answer : ℝ) s.t.
: (0 < answer ∧ answer < (4 * Real.sqrt 2) / 9 ∧
    (∫ x in Set.Ioo 0 ((Real.sqrt 2) / 3), max (answer - (2 * x - 3 * x ^ 3)) 0) =
    (∫ x in Set.Ioo 0 ((Real.sqrt 6) / 3), max ((2 * x - 3 * x ^ 3) - answer) 0))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q152
find_all (answer : ℝ) iff
: (0 < answer ∧ answer < (4 * Real.sqrt 2) / 9 ∧
    (∫ x in Set.Ioo 0 ((Real.sqrt 2) / 3), max (answer - (2 * x - 3 * x ^ 3)) 0) =
    (∫ x in Set.Ioo 0 ((Real.sqrt 6) / 3), max ((2 * x - 3 * x ^ 3) - answer) 0))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
