import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Convert $\frac{57}{160}$ to a terminating decimal.

## Answer
.35625
-/

-- # Formal Answer
noncomputable abbrev Q277.gt := fun (answer : ℝ) => answer = 0.35625

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q277
find (answer : ℝ) s.t.
: (answer = (57:ℚ) / 160)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q277
find_all (answer : ℝ) iff
: (answer = (57:ℚ) / 160)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
