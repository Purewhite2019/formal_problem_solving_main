import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose that $a$ and $b$ are nonzero real numbers, and that the [[equation]] $x^2 + ax + b = 0$ has solutions $a$ and $b$. Then the pair $(a,b)$ is

$\mathrm{(A)}\ (-2,1)
\qquad\mathrm{(B)}\ (-1,2)
\qquad\mathrm{(C)}\ (1,-2)
\qquad\mathrm{(D)}\ (2,-1)
\qquad\mathrm{(E)}\ (4,4)$

## Answer
(1, -2)
-/

-- # Formal Answer
noncomputable abbrev Q63.gt := fun (answer : ℝ × ℝ) => answer = (((1 : ℝ), (-2 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q63
find (answer : ℝ × ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁ : ∀ x, x^2 + a * x + b = (x - a) * (x - b))
: (answer = (a, b))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q63
find_all (answer : ℝ × ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁ : ∀ x, x^2 + a * x + b = (x - a) * (x - b))
: (answer = (a, b))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
