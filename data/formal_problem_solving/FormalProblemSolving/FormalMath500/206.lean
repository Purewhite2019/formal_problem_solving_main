import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
In the equation $|x-4| -10 = 2$, what is the product of all possible values of $x$?

## Answer
-128
-/

-- # Formal Answer
noncomputable abbrev Q206.gt := fun (answer : ℝ) => answer = (-128 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q206
find (answer : ℝ) s.t.
  (x₁ : ℝ)
  (x₂ : ℝ)
  (hx : x₁ ≠ x₂)
  (h₁ : |x₁ - 4| - 10 = 2 ∧ |x₂ - 4| - 10 = 2)
: (answer = x₁ * x₂)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q206
find_all (answer : ℝ) iff
  (x₁ : ℝ)
  (x₂ : ℝ)
  (hx : x₁ ≠ x₂)
  (h₁ : |x₁ - 4| - 10 = 2 ∧ |x₂ - 4| - 10 = 2)
: (answer = x₁ * x₂)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
