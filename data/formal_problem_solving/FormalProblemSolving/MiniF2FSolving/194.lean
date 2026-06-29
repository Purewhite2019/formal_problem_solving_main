import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the units digit of the sum of the squares of the first nine positive integers?

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q194.gt := fun (answer : ℤ) => answer = (5 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q194
find (answer : ℤ) s.t.
: (answer = (∑ x in Finset.Icc 1 9, x^2) % 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q194
find_all (answer : ℤ) iff
: (answer = (∑ x in Finset.Icc 1 9, x^2) % 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
