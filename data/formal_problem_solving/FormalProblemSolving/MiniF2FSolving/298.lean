import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Determine the remainder of 1529 (mod 6).

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q298.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q298
find (answer : ℕ) s.t.
: (answer = 1529 % 6)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q298
find_all (answer : ℕ) iff
: (answer = 1529 % 6)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
