import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Convert $\frac{21}{2^2 \cdot 5^7}$ to a terminating decimal.

## Answer
.0000672
-/

-- # Formal Answer
noncomputable abbrev Q290.gt := fun (answer : ℝ) => answer = 672e-7

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q290
find (answer : ℝ) s.t.
: (answer = (21:ℚ) / (2^2 * 5^7))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q290
find_all (answer : ℝ) iff
: (answer = (21:ℚ) / (2^2 * 5^7))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
