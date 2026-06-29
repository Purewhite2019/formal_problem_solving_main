import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve for $x$: $2^{2x} = 256^\frac{1}{2}$.

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q137.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q137
find (answer : ℝ) s.t.
: ((2 : ℝ) ^ (2 * answer) = (256 : ℝ) ^ (1/2 : ℝ))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q137
find_all (answer : ℝ) iff
: ((2 : ℝ) ^ (2 * answer) = (256 : ℝ) ^ (1/2 : ℝ))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
