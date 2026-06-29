import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Determine the modulo 4 remainder of the following sum: $$ 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12. $$

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q329.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q329
find (answer : ℕ) s.t.
: (answer = (∑ k in Finset.Icc 1 12, k) % 4)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q329
find_all (answer : ℕ) iff
: (answer = (∑ k in Finset.Icc 1 12, k) % 4)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
