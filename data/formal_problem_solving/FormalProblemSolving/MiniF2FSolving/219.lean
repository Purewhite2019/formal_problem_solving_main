import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many integers between 15 and 85 are divisible by 20?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q219.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q219
find (answer : ℕ) s.t.
  (S : Set ℤ)
  (hS : S = {x ∈ Set.Icc 15 85 | 20 ∣ x})
: (Nat.card S = answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q219
find_all (answer : ℕ) iff
  (S : Set ℤ)
  (hS : S = {x ∈ Set.Icc 15 85 | 20 ∣ x})
: (Nat.card S = answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
