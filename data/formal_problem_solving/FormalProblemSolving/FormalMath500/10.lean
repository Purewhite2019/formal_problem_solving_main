import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
You have 5 shirts, 6 pairs of pants, and 8 hats.  How many outfits can you make consisting of one shirt, one pair of pants, and one hat?

## Answer
240
-/

-- # Formal Answer
noncomputable abbrev Q10.gt := fun (answer : ℕ) => answer = (240 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q10
find (answer : ℕ) s.t.
  (shirts : ℕ)
  (pants : ℕ)
  (hats : ℕ)
  (h₀ : shirts = 5)
  (h₁ : pants = 6)
  (h₂ : hats = 8)
: (shirts * pants * hats = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q10
find_all (answer : ℕ) iff
  (shirts : ℕ)
  (pants : ℕ)
  (hats : ℕ)
  (h₀ : shirts = 5)
  (h₁ : pants = 6)
  (h₂ : hats = 8)
: (shirts * pants * hats = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
