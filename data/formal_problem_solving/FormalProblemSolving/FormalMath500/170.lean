import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the difference between the positive square root of 64 and the cube root of 64?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q170.gt := fun (answer : ℝ) => answer = (4 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q170
find (answer : ℝ) s.t.
: (Real.sqrt 64 - Real.rpow 64 ((1:ℚ) / 3) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q170
find_all (answer : ℝ) iff
: (Real.sqrt 64 - Real.rpow 64 ((1:ℚ) / 3) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
