import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For what real values of $x$ is $-4<x^{4}+4x^{2}<21$ satisfied? Express your answer in interval notation.

## Answer
(-\sqrt{3}, \sqrt{3})
-/

-- # Formal Answer
noncomputable abbrev Q163.gt := fun (answer : Set ℝ) => answer = (Set.Ioo (-(√(3 : ℝ) : ℝ) : ℝ) (√(3 : ℝ) : ℝ) : Set ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q163
find (answer : Set ℝ) s.t.
  (x : ℝ)
: (answer = {x:ℝ | -4 < x^4 + 4 * x^2 ∧ x^4 + 4 * x^2 < 21})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q163
find_all (answer : Set ℝ) iff
  (x : ℝ)
: (answer = {x:ℝ | -4 < x^4 + 4 * x^2 ∧ x^4 + 4 * x^2 < 21})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
