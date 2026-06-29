import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many positive three-digit integers are congruent to $7 \pmod{19}?$

## Answer
48
-/

-- # Formal Answer
noncomputable abbrev Q67.gt := fun (answer : ℕ) => answer = (48 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q67
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {x ∈ Set.Icc 100 999 | x % 19 = 7})
: (answer = Finset.card (Finset.filter (λ x => x % 19 = 7) (Finset.Icc 100 999)))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q67
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {x ∈ Set.Icc 100 999 | x % 19 = 7})
: (answer = Finset.card (Finset.filter (λ x => x % 19 = 7) (Finset.Icc 100 999)))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
