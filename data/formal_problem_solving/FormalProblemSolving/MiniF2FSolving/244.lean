import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Convert $852_9$ to base 10.

## Answer
695
-/

-- # Formal Answer
noncomputable abbrev Q244.gt := fun (answer : ℕ) => answer = (695 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q244
find (answer : ℕ) s.t.
: (answer = 8 * 9^2 + 5 * 9 + 2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q244
find_all (answer : ℕ) iff
: (answer = 8 * 9^2 + 5 * 9 + 2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
