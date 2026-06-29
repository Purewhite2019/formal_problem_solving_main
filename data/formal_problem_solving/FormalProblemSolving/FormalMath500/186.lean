import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Fake gold bricks are made by covering concrete cubes with gold paint, so the cost of the paint is proportional to their surface area while the cost of the concrete is proportional to their volume.  If a 1 inch cube costs $\$1.30$ to make while a 2 inch cube costs $\$6.80$, then how much would a 3 inch cube cost?

## Answer
\$18.90
-/

-- # Formal Answer
noncomputable abbrev Q186.gt := fun (answer : ℝ) => answer = 18.90

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q186
find (answer : ℝ) s.t.
  (cost : ℝ → ℝ)
  (k_paint : ℝ)
  (k_concrete : ℝ)
  (h₀ : ∀ (s : ℝ), cost s = k_paint * (6 * s^2) + k_concrete * (s^3))
  (h₁ : cost 1 = 1.30)
  (h₂ : cost 2 = 6.80)
: (cost 3 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q186
find_all (answer : ℝ) iff
  (cost : ℝ → ℝ)
  (k_paint : ℝ)
  (k_concrete : ℝ)
  (h₀ : ∀ (s : ℝ), cost s = k_paint * (6 * s^2) + k_concrete * (s^3))
  (h₁ : cost 1 = 1.30)
  (h₂ : cost 2 = 6.80)
: (cost 3 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
