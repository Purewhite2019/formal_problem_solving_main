import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the smallest real number $x$ in the domain of the function $$g(x) = \sqrt{(x-3)^2-(x-8)^2}~?$$

## Answer
\frac{11}{2}
-/

-- # Formal Answer
noncomputable abbrev Q133.gt := fun (answer : ℝ) => answer = ((↑(11 : ℚ) : ℝ) / (2 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q133
find (answer : ℝ) s.t.
  (x : ℝ)
  (h_domain : (x - 3)^2 - (x - 8)^2 ≥ 0)
  (h_min : ∀ y : ℝ, (y - 3)^2 - (y - 8)^2 ≥ 0 → x ≤ y)
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q133
find_all (answer : ℝ) iff
  (x : ℝ)
  (h_domain : (x - 3)^2 - (x - 8)^2 ≥ 0)
  (h_min : ∀ y : ℝ, (y - 3)^2 - (y - 8)^2 ≥ 0 → x ≤ y)
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
