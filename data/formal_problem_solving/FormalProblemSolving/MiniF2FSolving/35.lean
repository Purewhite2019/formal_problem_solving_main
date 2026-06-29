import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For what values of $x$ is it true that $x^2 - 5x - 4 \le 10$? Express your answer in interval notation.

## Answer
x \in [-2,7]
-/

-- # Formal Answer
noncomputable abbrev Q35.gt := fun (answer : Set ℝ) => answer = (Set.Icc (-2 : ℝ) (7 : ℝ) : Set ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q35
find (answer : Set ℝ) s.t.
  (x : ℝ)
: (answer = {x : ℝ| x^2 - 5 * x - 4 ≤ 10})
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q35
find_all (answer : Set ℝ) iff
  (x : ℝ)
: (answer = {x : ℝ| x^2 - 5 * x - 4 ≤ 10})
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
