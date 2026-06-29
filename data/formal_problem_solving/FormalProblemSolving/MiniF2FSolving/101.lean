import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $\log (xy^3) = 1$ and $\log (x^2y) = 1$, what is $\log (xy)$?

$\mathrm{(A)}\ -\frac 12
\qquad\mathrm{(B)}\ 0
\qquad\mathrm{(C)}\ \frac 12
\qquad\mathrm{(D)}\ \frac 35
\qquad\mathrm{(E)}\ 1$

## Answer
$\frac{3}{5}$
-/

-- # Formal Answer
noncomputable abbrev Q101.gt := fun (answer : ℝ) => answer = (3 / 5 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q101
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : 0 < x ∧ 0 < y)
  (h₁ : Real.log (x * y^3) = 1)
  (h₂ : Real.log (x^2 * y) = 1)
: (answer = Real.log (x * y))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q101
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : 0 < x ∧ 0 < y)
  (h₁ : Real.log (x * y^3) = 1)
  (h₂ : Real.log (x^2 * y) = 1)
: (answer = Real.log (x * y))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
