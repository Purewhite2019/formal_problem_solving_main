import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What power of 4 is equal to 8? Express your answer as a common fraction.

## Answer
\frac{3}{2}
-/

-- # Formal Answer
noncomputable abbrev Q201.gt := fun (answer : ℝ) => answer = (3 / 2 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q201
find (answer : ℝ) s.t.
: ((4:ℝ) ^ answer = 8)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q201
find_all (answer : ℝ) iff
: ((4:ℝ) ^ answer = 8)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
