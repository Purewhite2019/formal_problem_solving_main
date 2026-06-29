import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Evaluate $\log_264$.

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q181.gt := fun (answer : ℝ) => answer = (6 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q181
find (answer : ℝ) s.t.
: (Real.logb 2 64 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q181
find_all (answer : ℝ) iff
: (Real.logb 2 64 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
