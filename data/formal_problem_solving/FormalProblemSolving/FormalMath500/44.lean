import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Write $\frac{3}{20}$ as a decimal.

## Answer
0.15
-/

-- # Formal Answer
noncomputable abbrev Q44.gt := fun (answer : ℝ) => answer = 0.15

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q44
find (answer : ℝ) s.t.
: ((3 / 20 : ℚ) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q44
find_all (answer : ℝ) iff
: ((3 / 20 : ℚ) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
