import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The smallest distance between the origin and a point on the graph of $y=\frac{1}{2}x^2-9$ can be expressed as $a$.  Find $a^2$.

## Answer
17
-/

-- # Formal Answer
noncomputable abbrev Q152.gt := fun (answer : ℝ) => answer = (17 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q152
find (answer : ℝ) s.t.
  (a : ℝ)
  (h_min_distance : ∀ (x : ℝ), Real.sqrt (x^2 + (0.5 * x^2 - 9)^2) ≥ a)
  (h_exists_min : ∃ (x : ℝ), Real.sqrt (x^2 + (0.5 * x^2 - 9)^2) = a)
: (a^2 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q152
find_all (answer : ℝ) iff
  (a : ℝ)
  (h_min_distance : ∀ (x : ℝ), Real.sqrt (x^2 + (0.5 * x^2 - 9)^2) ≥ a)
  (h_exists_min : ∃ (x : ℝ), Real.sqrt (x^2 + (0.5 * x^2 - 9)^2) = a)
: (a^2 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
