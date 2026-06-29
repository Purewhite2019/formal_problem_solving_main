import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The volume of a cone is given by the formula $V = \frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume?

## Answer
65
-/

-- # Formal Answer
noncomputable abbrev Q190.gt := fun (answer : ℝ) => answer = (65 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q190
find (answer : ℝ) s.t.
  (b : ℝ)
  (h : ℝ)
  (v : ℝ)
  (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)
  (h₁ : v = 1 / 3 * (b * h))
  (h₂ : b = 30)
  (h₃ : h = 13 / 2)
: (answer = v)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q190
find_all (answer : ℝ) iff
  (b : ℝ)
  (h : ℝ)
  (v : ℝ)
  (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)
  (h₁ : v = 1 / 3 * (b * h))
  (h₂ : b = 30)
  (h₃ : h = 13 / 2)
: (answer = v)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
