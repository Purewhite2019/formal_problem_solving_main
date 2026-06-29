import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If a snack-size tin of peaches has $40$ calories and is $2\%$ of a person's daily caloric requirement, how many calories fulfill a person's daily caloric requirement?

## Answer
2000
-/

-- # Formal Answer
noncomputable abbrev Q214.gt := fun (answer : ℝ) => answer = (2000 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q214
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : 0.02 * x = 40)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q214
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : 0.02 * x = 40)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
