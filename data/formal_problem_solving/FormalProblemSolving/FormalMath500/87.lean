import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The area of triangle $ABC$ is equal to $a^2 - (b - c)^2,$ where $a,$ $b,$ and $c$ are the sides of triangle $ABC,$ as usual.  Compute $\tan A.$

## Answer
\frac{8}{15}
-/

-- # Formal Answer
noncomputable abbrev Q87.gt := fun (answer : ℝ) => answer = (8 / 15 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q87
find (answer : ℝ) s.t.
  (area : ℝ)
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (A : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < A ∧ A < π)
  (h_area1 : area = a^2 - (b - c)^2)
  (h_area2 : area = (b * c * Real.sin A) / 2)
  (h_cos : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
: (Real.tan A = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q87
find_all (answer : ℝ) iff
  (area : ℝ)
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (A : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hA : 0 < A ∧ A < π)
  (h_area1 : area = a^2 - (b - c)^2)
  (h_area2 : area = (b * c * Real.sin A) / 2)
  (h_cos : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
: (Real.tan A = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
