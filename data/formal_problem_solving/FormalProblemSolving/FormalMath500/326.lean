import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x) = x^3 + 3x^2 + 1.$  There exist real numbers $a \neq 0$ and $b,$ such that
\[f(x) - f(a) = (x - a)^2 (x - b).\]Enter the ordered pair $(a,b).$

## Answer
(-2,1)
-/

-- # Formal Answer
noncomputable abbrev Q326.gt := fun (answer : ℝ × ℝ) => answer = (((-2 : ℝ), (1 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q326
find (answer : ℝ × ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = x^3 + 3 * x^2 + 1)
  (a : ℝ)
  (b : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : ∀ x, f x - f a = (x - a)^2 * (x - b))
: ((a, b) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q326
find_all (answer : ℝ × ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = x^3 + 3 * x^2 + 1)
  (a : ℝ)
  (b : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : ∀ x, f x - f a = (x - a)^2 * (x - b))
: ((a, b) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
