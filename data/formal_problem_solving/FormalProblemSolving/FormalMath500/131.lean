import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f$ be defined by  \[f(x) = \left\{
\begin{array}{cl}
3-x & \text{ if } x \leq 3, \\
-x^3+2x^2+3x & \text{ if } x>3.
\end{array}
\right.\]Calculate $f^{-1}(0)+f^{-1}(6)$.

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q131.gt := fun (answer : ℝ) => answer = (0 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q131
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ x, x ≤ 3 → f x = 3 - x)
  (h₁ : ∀ x, x > 3 → f x = -x ^ 3 + 2 * x ^ 2 + 3 * x)
: (answer = Function.invFun f 0 + Function.invFun f 6)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q131
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ x, x ≤ 3 → f x = 3 - x)
  (h₁ : ∀ x, x > 3 → f x = -x ^ 3 + 2 * x ^ 2 + 3 * x)
: (answer = Function.invFun f 0 + Function.invFun f 6)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
