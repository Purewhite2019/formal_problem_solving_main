import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the units digit of the product of all of the odd integers between 0 and 12?

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q257.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q257
find (answer : ℕ) s.t.
  (S : Finset ℕ)
  (hS : S = Finset.range 6)
: (answer = (∏ k in S, (2 * k + 1)) % 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q257
find_all (answer : ℕ) iff
  (S : Finset ℕ)
  (hS : S = Finset.range 6)
: (answer = (∏ k in S, (2 * k + 1)) % 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
