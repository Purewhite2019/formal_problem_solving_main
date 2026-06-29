import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $p(x)$ be a quadratic polynomial with integer coefficients which has $4-\sqrt{11}$ as a root. Compute $\frac{p(3)}{p(4)}.$

## Answer
\frac{10}{11}
-/

-- # Formal Answer
noncomputable abbrev Q296.gt := fun (answer : ℝ) => answer = (10 / 11 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q296
find (answer : ℝ) s.t.
  (a : ℤ)
  (b : ℤ)
  (c : ℤ)
  (p : ℝ → ℝ)
  (h_quad : a ≠ 0)
  (h_p : ∀ x, p x = (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ))
  (h_root : p (4 - Real.sqrt 11) = 0)
: (answer = p 3 / p 4)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q296
find_all (answer : ℝ) iff
  (a : ℤ)
  (b : ℤ)
  (c : ℤ)
  (p : ℝ → ℝ)
  (h_quad : a ≠ 0)
  (h_p : ∀ x, p x = (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ))
  (h_root : p (4 - Real.sqrt 11) = 0)
: (answer = p 3 / p 4)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
