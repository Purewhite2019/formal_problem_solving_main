import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
In a convex quadrilateral, the measure of the largest angle is twice the measure of the smallest angle, and the other two angles are both right angles. How many degrees are in the largest angle?

## Answer
120
-/

-- # Formal Answer
noncomputable abbrev Q39.gt := fun (answer : ℝ) => answer = (120 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q39
find (answer : ℝ) s.t.
  (smallest_angle : ℝ)
  (largest_angle : ℝ)
  (sum_of_angles : ℝ)
  (h₀ : largest_angle = 2 * smallest_angle)
  (h₁ : sum_of_angles = 360)
  (h₂ : sum_of_angles = largest_angle + smallest_angle + 90 + 90)
: (largest_angle = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q39
find_all (answer : ℝ) iff
  (smallest_angle : ℝ)
  (largest_angle : ℝ)
  (sum_of_angles : ℝ)
  (h₀ : largest_angle = 2 * smallest_angle)
  (h₁ : sum_of_angles = 360)
  (h₂ : sum_of_angles = largest_angle + smallest_angle + 90 + 90)
: (largest_angle = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
