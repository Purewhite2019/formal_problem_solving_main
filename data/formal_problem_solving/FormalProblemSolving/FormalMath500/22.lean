import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many sides would there be in a convex polygon if the sum of all but one of its interior angles is $1070^{\circ}$?

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q22.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q22
find (answer : ℕ) s.t.
: (3 ≤ answer ∧ ∃ x : ℕ, 0 < x ∧ x < 180 ∧ (answer - 2) * 180 = 1070 + x)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q22
find_all (answer : ℕ) iff
: (3 ≤ answer ∧ ∃ x : ℕ, 0 < x ∧ x < 180 ∧ (answer - 2) * 180 = 1070 + x)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
