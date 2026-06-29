import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute $29^{13} - 5^{13}$ modulo 7.

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q234.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q234
find (answer : ℕ) s.t.
: (answer = (29^13 - 5^13) % 7)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q234
find_all (answer : ℕ) iff
: (answer = (29^13 - 5^13) % 7)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
