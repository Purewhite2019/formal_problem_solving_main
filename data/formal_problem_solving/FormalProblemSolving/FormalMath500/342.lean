import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $P(x)$ be a quadratic polynomial with real coefficients satisfying $x^2 - 2x + 2 \le P(x) \le 2x^2 - 4x + 3$ for all real  numbers $x$, and suppose $P(11) = 181$. Find $P(16)$.

## Answer
406
-/

-- # Formal Answer
noncomputable abbrev Q342.gt := fun (answer : ℝ) => answer = (406 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q342
find (answer : ℝ) s.t.
  (P : ℝ → ℝ)
  (h_quadratic : ∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c)
  (h_bounds : ∀ x, x^2 - 2 * x + 2 ≤ P x ∧ P x ≤ 2 * x^2 - 4 * x + 3)
  (h_P11 : P 11 = 181)
: (P 16 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q342
find_all (answer : ℝ) iff
  (P : ℝ → ℝ)
  (h_quadratic : ∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c)
  (h_bounds : ∀ x, x^2 - 2 * x + 2 ≤ P x ∧ P x ≤ 2 * x^2 - 4 * x + 3)
  (h_P11 : P 11 = 181)
: (P 16 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
