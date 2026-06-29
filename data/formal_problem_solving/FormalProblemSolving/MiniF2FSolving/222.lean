import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the remainder when 5 to the 999,999th power is divided by 7?

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q222.gt := fun (answer : ℕ) => answer = (6 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q222
find (answer : ℕ) s.t.
: (answer = (5^999999) % 7)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q222
find_all (answer : ℕ) iff
: (answer = (5^999999) % 7)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
