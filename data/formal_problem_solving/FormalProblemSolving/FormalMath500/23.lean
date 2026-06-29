import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
In a certain isosceles right triangle, the altitude to the hypotenuse has length $4\sqrt{2}$.  What is the area of the triangle?

## Answer
32
-/

-- # Formal Answer
noncomputable abbrev Q23.gt := fun (answer : ℝ) => answer = (32 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q23
find (answer : ℝ) s.t.
  (a : ℝ)
  (h : ℝ)
  (h_leg_pos : 0 < a)
  (h_altitude_eq : h = 4 * Real.sqrt 2)
  (h_area_condition : h * (a * Real.sqrt 2) = a * a)
: (answer = a ^ 2 / 2)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q23
find_all (answer : ℝ) iff
  (a : ℝ)
  (h : ℝ)
  (h_leg_pos : 0 < a)
  (h_altitude_eq : h = 4 * Real.sqrt 2)
  (h_area_condition : h * (a * Real.sqrt 2) = a * a)
: (answer = a ^ 2 / 2)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
