import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the greatest common factor of 180 and 168.

## Answer
12
-/

-- # Formal Answer
noncomputable abbrev Q150.gt := fun (answer : ℕ) => answer = (12 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q150
find (answer : ℕ) s.t.
: (answer = Nat.gcd 180 168)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q150
find_all (answer : ℕ) iff
: (answer = Nat.gcd 180 168)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
