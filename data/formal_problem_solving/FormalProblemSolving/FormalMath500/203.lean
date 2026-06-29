import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x) = x - 3$ and $q(x) = bx +1$.  If $f(q(1)) = -3$, what is $b$?

## Answer
-1
-/

-- # Formal Answer
noncomputable abbrev Q203.gt := fun (answer : ℝ) => answer = (-1 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q203
find (answer : ℝ) s.t.
  (b : ℝ)
  (f : ℝ → ℝ)
  (q : ℝ → ℝ)
  (h₀ : ∀ x, f x = x - 3)
  (h₁ : ∀ x, q x = b * x + 1)
  (h₂ : f (q 1) = -3)
: (b = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q203
find_all (answer : ℝ) iff
  (b : ℝ)
  (f : ℝ → ℝ)
  (q : ℝ → ℝ)
  (h₀ : ∀ x, f x = x - 3)
  (h₁ : ∀ x, q x = b * x + 1)
  (h₂ : f (q 1) = -3)
: (b = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
