import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute $a+b+c,$ given that $a,$ $b,$ and $c$ are the roots of \[\frac{1}{x} + 5x^2 = 6x - 24.\]

## Answer
\frac65
-/

-- # Formal Answer
noncomputable abbrev Q369.gt := fun (answer : ℂ) => answer = (6 / 5 : ℂ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q369
find (answer : ℂ) s.t.
  (a : ℂ)
  (b : ℂ)
  (c : ℂ)
  (h_poly : ∀ x : ℂ, 5 * x ^ 3 - 6 * x ^ 2 + 24 * x + 1 = 5 * (x - a) * (x - b) * (x - c))
: (a + b + c = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q369
find_all (answer : ℂ) iff
  (a : ℂ)
  (b : ℂ)
  (c : ℂ)
  (h_poly : ∀ x : ℂ, 5 * x ^ 3 - 6 * x ^ 2 + 24 * x + 1 = 5 * (x - a) * (x - b) * (x - c))
: (a + b + c = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
