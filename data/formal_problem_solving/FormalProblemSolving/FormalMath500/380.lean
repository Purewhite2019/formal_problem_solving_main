import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute the product of the number $5+\sqrt{3}$ and its radical conjugate.

## Answer
22
-/

-- # Formal Answer
noncomputable abbrev Q380.gt := fun (answer : ℝ) => answer = (22 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q380
find (answer : ℝ) s.t.
: ((5 + Real.sqrt 3) * (5 - Real.sqrt 3) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q380
find_all (answer : ℝ) iff
: ((5 + Real.sqrt 3) * (5 - Real.sqrt 3) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
