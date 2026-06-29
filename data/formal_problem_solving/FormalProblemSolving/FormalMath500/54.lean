import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The lengths of two opposite sides of a square are decreased by $40\%$ while the lengths of the other two sides are increased by $50\%$ to form a rectangle. By what percent does the square's area decrease?

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q54.gt := fun (answer : ℝ) => answer = (10 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q54
find (answer : ℝ) s.t.
  (s : ℝ)
  (original_area : ℝ)
  (new_area : ℝ)
  (h_pos : s > 0)
  (h_original_area : s * s = original_area)
  (h_new_area : (0.6 * s) * (1.5 * s) = new_area)
: (((original_area - new_area) / original_area) * 100 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q54
find_all (answer : ℝ) iff
  (s : ℝ)
  (original_area : ℝ)
  (new_area : ℝ)
  (h_pos : s > 0)
  (h_original_area : s * s = original_area)
  (h_new_area : (0.6 * s) * (1.5 * s) = new_area)
: (((original_area - new_area) / original_area) * 100 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
