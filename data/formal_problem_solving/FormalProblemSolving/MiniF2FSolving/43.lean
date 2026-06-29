import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the result when the greatest common factor of 6432 and 132 is increased by 11?

## Answer
23
-/

-- # Formal Answer
noncomputable abbrev Q43.gt := fun (answer : ℕ) => answer = (23 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q43
find (answer : ℕ) s.t.
: (answer = (Nat.gcd 6432 132) + 11)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q43
find_all (answer : ℕ) iff
: (answer = (Nat.gcd 6432 132) + 11)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
