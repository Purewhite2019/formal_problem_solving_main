import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve for $x: 3^{2x} + 19 = 10^x$.

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q303.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q303
find (answer : ℝ) s.t.
: ((3:ℝ)^(2*answer) + 19 = (10:ℝ)^answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q303
find_all (answer : ℝ) iff
: ((3:ℝ)^(2*answer) + 19 = (10:ℝ)^answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
