import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If a snack-size tin of peaches has $40$ calories and is $2\%$ of a person's daily caloric requirement, how many calories fulfill a person's daily caloric requirement?

## Answer
2000
-/

-- # Formal Answer
noncomputable abbrev Q116.gt := fun (answer : ℕ) => answer = (2000 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q116
find (answer : ℕ) s.t.
  (calories_per_tin : ℕ)
  (percentage_of_daily : ℝ)
  (h₀ : calories_per_tin = 40)
  (h₁ : percentage_of_daily = 0.02)
: (calories_per_tin / percentage_of_daily = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q116
find_all (answer : ℕ) iff
  (calories_per_tin : ℕ)
  (percentage_of_daily : ℝ)
  (h₀ : calories_per_tin = 40)
  (h₁ : percentage_of_daily = 0.02)
: (calories_per_tin / percentage_of_daily = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
