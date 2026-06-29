import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If each point of the circle $x^2 + y^2 = 25$ is reflected in the point $(4,1),$ the set of image points satisfies the equation
\[x^2 + ay^2 + bx + cy + d = 0.\]Compute the ordered quadruple $(a,b,c,d)$ of real numbers.

## Answer
(1,-16,-4,43)
-/

-- # Formal Answer
noncomputable abbrev Q301.gt := fun (answer : ℝ × ℝ × ℝ × ℝ) => answer = (((1 : ℝ), (((-16 : ℝ), (((-4 : ℝ), (43 : ℝ)) : ℝ × ℝ)) : ℝ × ℝ × ℝ)) : ℝ × ℝ × ℝ × ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q301
find (answer : ℝ × ℝ × ℝ × ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (h_reflection : ∀ (x y : ℝ), x^2 + y^2 = 25 → ∃ (x' y' : ℝ), (x' = 2 * 4 - x ∧ y' = 2 * 1 - y) ∧ (x'^2 + a * y'^2 + b * x' + c * y' + d = 0))
: ((a, b, c, d) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q301
find_all (answer : ℝ × ℝ × ℝ × ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (h_reflection : ∀ (x y : ℝ), x^2 + y^2 = 25 → ∃ (x' y' : ℝ), (x' = 2 * 4 - x ∧ y' = 2 * 1 - y) ∧ (x'^2 + a * y'^2 + b * x' + c * y' + d = 0))
: ((a, b, c, d) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
