import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Determine the remainder of 71 (mod 3).

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q66.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q66
find (answer : ℕ) s.t.
: (answer = 71 % 3)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q66
find_all (answer : ℕ) iff
: (answer = 71 % 3)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
