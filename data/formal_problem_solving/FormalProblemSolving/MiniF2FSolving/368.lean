import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the smallest positive integer, other than $1$, that is both a perfect cube and a perfect fourth power?

## Answer
4096
-/

-- # Formal Answer
noncomputable abbrev Q368.gt := fun (answer : ℕ) => answer = (4096 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q368
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {n | n > 1 ∧ (∃ x, x^3 = n) ∧ (∃ t, t^4 = n)})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q368
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {n | n > 1 ∧ (∃ x, x^3 = n) ∧ (∃ t, t^4 = n)})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
