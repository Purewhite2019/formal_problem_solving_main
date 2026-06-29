import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the value of $(26^2 - 24^2 - 10)^2 - 10^2$?

## Answer
8000
-/

-- # Formal Answer
noncomputable abbrev Q155.gt := fun (answer : ℝ) => answer = (8000 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q155
find (answer : ℝ) s.t.
: ((26^2 - 24^2 - 10)^2 - 10^2 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q155
find_all (answer : ℝ) iff
: ((26^2 - 24^2 - 10)^2 - 10^2 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
