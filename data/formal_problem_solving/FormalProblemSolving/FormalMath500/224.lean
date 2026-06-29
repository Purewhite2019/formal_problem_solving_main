import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Twelve friends met for dinner at Oscar's Overstuffed Oyster House, and each ordered one meal. The portions were so large, there was enough food for 18 people. If they share, how many meals should they have ordered to have just enough food for the 12 of them?

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q224.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q224
find (answer : ℕ) s.t.
  (friends : ℕ)
  (meals_ordered : ℕ)
  (total_food : ℕ)
  (meals_needed : ℕ)
  (h₀ : friends = 12)
  (h₁ : meals_ordered = 12)
  (h₂ : total_food = 18)
  (h₃ : meals_needed = friends * meals_ordered / total_food)
: (meals_needed = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q224
find_all (answer : ℕ) iff
  (friends : ℕ)
  (meals_ordered : ℕ)
  (total_food : ℕ)
  (meals_needed : ℕ)
  (h₀ : friends = 12)
  (h₁ : meals_ordered = 12)
  (h₂ : total_food = 18)
  (h₃ : meals_needed = friends * meals_ordered / total_food)
: (meals_needed = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
