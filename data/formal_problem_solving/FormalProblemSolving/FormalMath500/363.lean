import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute: $1-2+3-4+5- \dots +99-100$.

## Answer
-50
-/

-- # Formal Answer
noncomputable abbrev Q363.gt := fun (answer : ℤ) => answer = (-50 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q363
find (answer : ℤ) s.t.
: (∑ k in Finset.range 100, (-1:ℤ)^k * (k + 1) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q363
find_all (answer : ℤ) iff
: (∑ k in Finset.range 100, (-1:ℤ)^k * (k + 1) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
