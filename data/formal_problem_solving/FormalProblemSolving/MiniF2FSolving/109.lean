import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the tens digit of $5^{2005}$?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q109.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q109
find (answer : ℕ) s.t.
: (answer = (5^2005) % 100 / 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q109
find_all (answer : ℕ) iff
: (answer = (5^2005) % 100 / 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
