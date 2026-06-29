import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x)$ be the polynomial \[f(x)=3x^4+5x^2-9x-2.\] If $g(x)$ is equal to the polynomial $f(x-1)$, what is the sum of the coefficients of $g$?

## Answer
-2
-/

-- # Formal Answer
noncomputable abbrev Q220.gt := fun (answer : ℝ) => answer = (-2 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q220
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f x = 3 * x ^ 4 + 5 * x ^ 2 - 9 * x - 2)
  (h₁ : ∀ x, g x = f (x - 1))
: (g 1 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q220
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f x = 3 * x ^ 4 + 5 * x ^ 2 - 9 * x - 2)
  (h₁ : ∀ x, g x = f (x - 1))
: (g 1 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
