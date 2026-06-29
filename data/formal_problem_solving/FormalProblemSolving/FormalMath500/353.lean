import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The asymptotes of a hyperbola are $y = 2x - 3$ and $y = 17 - 2x.$  Also, the hyperbola passes through the point $(4,7).$  Find the distance between the foci of the hyperbola.

## Answer
2 \sqrt{5}
-/

-- # Formal Answer
noncomputable abbrev Q353.gt := fun (answer : ℝ) => answer = ((2 : ℝ) * (√(5 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q353
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h : ℝ)
  (k : ℝ)
  (hk : (4 - h)^2 / (a^2) - (7 - k)^2 / (b^2) = 1)
  (hab : b / a = 2)
  (hk2 : 2 * h  - 3 = 17 - 2*h ∧ k = 2*h-3)
: (answer = 2 * Real.sqrt (a^2 + b^2))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q353
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h : ℝ)
  (k : ℝ)
  (hk : (4 - h)^2 / (a^2) - (7 - k)^2 / (b^2) = 1)
  (hab : b / a = 2)
  (hk2 : 2 * h  - 3 = 17 - 2*h ∧ k = 2*h-3)
: (answer = 2 * Real.sqrt (a^2 + b^2))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
