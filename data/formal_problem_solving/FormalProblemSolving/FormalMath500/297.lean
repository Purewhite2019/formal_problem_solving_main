import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
An equilateral triangle is inscribed in the parabola $x^2 = 8y,$ such that one of the vertices of the triangle coincides with the vertex of the parabola.  Find the side length of this equilateral triangle.

## Answer
16 \sqrt{3}
-/

-- # Formal Answer
noncomputable abbrev Q297.gt := fun (answer : ℝ) => answer = ((16 : ℝ) * (√(3 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q297
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (hxy : x > 0 ∧ Real.sqrt (x^2 + y^2) = 2*x ∧ x^2 = 8 * y)
: (answer = 2*x)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q297
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (hxy : x > 0 ∧ Real.sqrt (x^2 + y^2) = 2*x ∧ x^2 = 8 * y)
: (answer = 2*x)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
