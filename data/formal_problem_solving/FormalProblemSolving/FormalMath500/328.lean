import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For some real numbers $a$ and $b$, the equation \[
8x^3 + 4ax^2 + 2bx + a = 0
\]has three distinct positive roots. If the sum of the base-2 logarithms of the roots is 5, what is the value of $a$?

## Answer
-256
-/

-- # Formal Answer
noncomputable abbrev Q328.gt := fun (answer : ℝ) => answer = (-256 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q328
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h : ∃ (x₁ x₂ x₃ : ℝ),
      x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧
      x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
      8 * x₁ ^ 3 + 4 * a * x₁ ^ 2 + 2 * b * x₁ + a = 0 ∧
      8 * x₂ ^ 3 + 4 * a * x₂ ^ 2 + 2 * b * x₂ + a = 0 ∧
      8 * x₃ ^ 3 + 4 * a * x₃ ^ 2 + 2 * b * x₃ + a = 0 ∧
      Real.logb 2 x₁ + Real.logb 2 x₂ + Real.logb 2 x₃ = 5)
: (answer = a)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q328
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h : ∃ (x₁ x₂ x₃ : ℝ),
      x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧
      x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
      8 * x₁ ^ 3 + 4 * a * x₁ ^ 2 + 2 * b * x₁ + a = 0 ∧
      8 * x₂ ^ 3 + 4 * a * x₂ ^ 2 + 2 * b * x₂ + a = 0 ∧
      8 * x₃ ^ 3 + 4 * a * x₃ ^ 2 + 2 * b * x₃ + a = 0 ∧
      Real.logb 2 x₁ + Real.logb 2 x₂ + Real.logb 2 x₃ = 5)
: (answer = a)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
