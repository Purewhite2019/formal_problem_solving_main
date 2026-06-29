import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the value of $(52+6\sqrt{43})^{3/2}-(52-6\sqrt{43})^{3/2}$.

## Answer
828
-/

-- # Formal Answer
noncomputable abbrev Q126.gt := fun (answer : ℝ) => answer = (828 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q126
find (answer : ℝ) s.t.
: (answer = (Real.sqrt (52 + 6 * Real.sqrt 43)) ^ 3 - (Real.sqrt (52 - 6 * Real.sqrt 43)) ^ 3)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q126
find_all (answer : ℝ) iff
: (answer = (Real.sqrt (52 + 6 * Real.sqrt 43)) ^ 3 - (Real.sqrt (52 - 6 * Real.sqrt 43)) ^ 3)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
