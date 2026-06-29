import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify $(u+4)(u-1) - (u-3)(u+6)$.

## Answer
14
-/

-- # Formal Answer
noncomputable abbrev Q189.gt := fun (answer : ℝ) => answer = (14 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q189
find (answer : ℝ) s.t.
  (u : ℝ)
: ((u + 4) * (u - 1) - (u - 3) * (u + 6) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q189
find_all (answer : ℝ) iff
  (u : ℝ)
: ((u + 4) * (u - 1) - (u - 3) * (u + 6) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
