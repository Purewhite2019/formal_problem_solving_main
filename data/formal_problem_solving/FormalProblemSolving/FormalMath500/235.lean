import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the remainder of 2004 divided by 12?

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q235.gt := fun (answer : ℝ) => answer = (0 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q235
find (answer : ℝ) s.t.
: (2004 % 12 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q235
find_all (answer : ℝ) iff
: (2004 % 12 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
