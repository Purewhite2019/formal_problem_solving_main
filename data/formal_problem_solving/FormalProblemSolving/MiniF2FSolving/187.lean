import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the units digit of $19^{19}+99^{99}$?

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q187.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q187
find (answer : ℕ) s.t.
: (answer = (19^19 + 99^99) % 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q187
find_all (answer : ℕ) iff
: (answer = (19^19 + 99^99) % 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
