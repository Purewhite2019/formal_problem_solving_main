import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For what values of $x$ is it true that $x^2 - 5x - 4 \le 10$? Express your answer in interval notation.

## Answer
x \in [-2,7]
-/

-- # Formal Answer
noncomputable abbrev Q229.gt := fun (answer : ℝ × ℝ) => answer = (((-2 : ℝ), (7 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q229
find (answer : ℝ × ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {x:ℝ | x^2 - 5*x - 14 ≤ 0})
: (Set.Icc answer.1 answer.2 = S)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q229
find_all (answer : ℝ × ℝ) iff
  (S : Set ℝ)
  (hS : S = {x:ℝ | x^2 - 5*x - 14 ≤ 0})
: (Set.Icc answer.1 answer.2 = S)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
