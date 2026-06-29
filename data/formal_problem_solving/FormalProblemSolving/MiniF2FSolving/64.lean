import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the remainder of $2^8$ when it is divided by 5.

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q64.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q64
find (answer : ℕ) s.t.
: (answer = (2^8) % 5)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q64
find_all (answer : ℕ) iff
: (answer = (2^8) % 5)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
