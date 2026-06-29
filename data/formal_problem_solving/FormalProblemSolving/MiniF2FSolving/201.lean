import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the ones digit of $1 \cdot 3 \cdot 5 \cdot 7 \cdot 9 \cdot 11 \cdot 13$?

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q201.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q201
find (answer : ℕ) s.t.
: (answer = (1 * 3 * 5 * 7 * 9 * 11 * 13) % 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q201
find_all (answer : ℕ) iff
: (answer = (1 * 3 * 5 * 7 * 9 * 11 * 13) % 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
