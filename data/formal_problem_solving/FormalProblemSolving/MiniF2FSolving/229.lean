import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the units digit of $2^{2010}$?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q229.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q229
find (answer : ℕ) s.t.
: (answer = (2^2010) % 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q229
find_all (answer : ℕ) iff
: (answer = (2^2010) % 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
