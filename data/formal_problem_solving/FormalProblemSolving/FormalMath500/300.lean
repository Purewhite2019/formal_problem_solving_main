import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f$ be the function defined by $f(x) = x^3 - 49x^2 + 623x - 2015,$ and let $g(x) = f(x + 5).$  Compute the sum of the roots of $g.$

## Answer
34
-/

-- # Formal Answer
noncomputable abbrev Q300.gt := fun (answer : ℝ) => answer = (34 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q300
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_f : ∀ x, f x = x^3 - 49 * x^2 + 623 * x - 2015)
  (h_g : ∀ x, g x = f (x + 5))
: (∃ (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧  g r₁ = 0 ∧ g r₂ = 0 ∧ g r₃ = 0 ∧ r₁ + r₂ + r₃ = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q300
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_f : ∀ x, f x = x^3 - 49 * x^2 + 623 * x - 2015)
  (h_g : ∀ x, g x = f (x + 5))
: (∃ (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧  g r₁ = 0 ∧ g r₂ = 0 ∧ g r₃ = 0 ∧ r₁ + r₂ + r₃ = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
