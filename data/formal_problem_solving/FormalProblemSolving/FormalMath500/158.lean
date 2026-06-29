import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the smallest number which is one less than twice its reciprocal?

## Answer
-2
-/

-- # Formal Answer
noncomputable abbrev Q158.gt := fun (answer : ℝ) => answer = (-2 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q158
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {x : ℝ | x = 2 * ((1:ℝ) / x) - 1})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q158
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {x : ℝ | x = 2 * ((1:ℝ) / x) - 1})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
