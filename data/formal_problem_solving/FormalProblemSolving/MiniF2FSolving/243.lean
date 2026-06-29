import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the smallest integer greater than 10 that is both a perfect square and a perfect cube?

## Answer
64
-/

-- # Formal Answer
noncomputable abbrev Q243.gt := fun (answer : ℕ) => answer = (64 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q243
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {n : ℕ | 10 < n ∧ (∃ x, x^2 = n) ∧ (∃ t, t^3 = n)})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q243
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {n : ℕ | 10 < n ∧ (∃ x, x^2 = n) ∧ (∃ t, t^3 = n)})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
