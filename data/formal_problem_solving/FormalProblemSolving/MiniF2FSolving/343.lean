import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Determine the remainder of 194 (mod 11).

## Answer
7
-/

-- # Formal Answer
noncomputable abbrev Q343.gt := fun (answer : ℕ) => answer = (7 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q343
find (answer : ℕ) s.t.
: (answer = 194 % 11)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q343
find_all (answer : ℕ) iff
: (answer = 194 % 11)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
