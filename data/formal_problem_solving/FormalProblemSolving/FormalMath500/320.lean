import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $p(x)$ be a polynomial of degree 5 such that
\[p(n) = \frac{n}{n^2 - 1}\]for $n = 2,$ 3, 4, $\dots,$ 7.  Find $p(8).$

## Answer
\frac{3}{56}
-/

-- # Formal Answer
noncomputable abbrev Q320.gt := fun (answer : ℚ) => answer = (3 / 56 : ℚ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q320
find (answer : ℚ) s.t.
  (p : ℝ → ℝ)
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (e : ℝ)
  (f : ℝ)
  (hp : ∀ x : ℝ, p x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f)
  (hp_vals : ∀ n ∈ [2,3,4,5,6,7], p (n : ℝ) = n / (n^2 - 1))
: (answer = p 8)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q320
find_all (answer : ℚ) iff
  (p : ℝ → ℝ)
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (e : ℝ)
  (f : ℝ)
  (hp : ∀ x : ℝ, p x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f)
  (hp_vals : ∀ n ∈ [2,3,4,5,6,7], p (n : ℝ) = n / (n^2 - 1))
: (answer = p 8)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
