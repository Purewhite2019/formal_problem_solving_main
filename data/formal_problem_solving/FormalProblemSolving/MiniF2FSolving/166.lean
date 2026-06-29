import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the remainder when 2003 is divided by 11?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q166.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q166
find (answer : ℕ) s.t.
: (answer = 2003 % 11)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q166
find_all (answer : ℕ) iff
: (answer = 2003 % 11)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
