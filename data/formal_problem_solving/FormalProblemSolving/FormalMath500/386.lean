import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $p(x)$ be a monic polynomial of degree 4.  Three of the roots of $p(x)$ are 1, 2, and 3.  Find $p(0) + p(4).$

## Answer
24
-/

-- # Formal Answer
noncomputable abbrev Q386.gt := fun (answer : ℝ) => answer = (24 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q386
find (answer : ℝ) s.t.
  (p : ℝ → ℝ)
  (a : ℝ)
  (h₀ : ∀ x, p x = (x - 1) * (x - 2) * (x - 3) * (x - a))
: (p 0 + p 4 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q386
find_all (answer : ℝ) iff
  (p : ℝ → ℝ)
  (a : ℝ)
  (h₀ : ∀ x, p x = (x - 1) * (x - 2) * (x - 3) * (x - a))
: (p 0 + p 4 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
