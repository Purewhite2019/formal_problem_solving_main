import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
In a 8 fluid ounce bottle of Vitamin Water, there are 125 calories. How many calories would be contained in a 12 fluid ounce bottle? Express your answer in decimal form.

## Answer
187.5
-/

-- # Formal Answer
noncomputable abbrev Q120.gt := fun (answer : ℝ) => answer = 187.5

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q120
find (answer : ℝ) s.t.
  (calories_per_ounce : ℝ)
  (h₀ : calories_per_ounce = 125 / 8)
: (answer = calories_per_ounce * 12)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q120
find_all (answer : ℝ) iff
  (calories_per_ounce : ℝ)
  (h₀ : calories_per_ounce = 125 / 8)
: (answer = calories_per_ounce * 12)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
