import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The smallest distance between the origin and a point on the graph of $y=\frac{1}{\sqrt{2}}\left(x^2-3\right)$ can be expressed as $\sqrt{a}/b$, where $a$ and $b$ are positive integers such that $a$ is not divisible by the square of any integer greater than one.  Find $a+b$.

## Answer
12
-/

-- # Formal Answer
noncomputable abbrev Q200.gt := fun (answer : ℕ) => answer = (12 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q200
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h_lower : ∀ x : ℝ, (x ^ 2 + (((1 : ℝ) / Real.sqrt 2) * (x ^ 2 - 3)) ^ 2) ≥ (a : ℝ) / ((b : ℝ) ^ 2))
  (h_exists : ∃ x : ℝ, (x ^ 2 + (((1 : ℝ) / Real.sqrt 2) * (x ^ 2 - 3)) ^ 2) = (a : ℝ) / ((b : ℝ) ^ 2))
  (h_sqfree : ∀ (k : ℕ), 2 ≤ k → ¬ (k ^ 2 ∣ a))
: (answer = a + b)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q200
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h_lower : ∀ x : ℝ, (x ^ 2 + (((1 : ℝ) / Real.sqrt 2) * (x ^ 2 - 3)) ^ 2) ≥ (a : ℝ) / ((b : ℝ) ^ 2))
  (h_exists : ∃ x : ℝ, (x ^ 2 + (((1 : ℝ) / Real.sqrt 2) * (x ^ 2 - 3)) ^ 2) = (a : ℝ) / ((b : ℝ) ^ 2))
  (h_sqfree : ∀ (k : ℕ), 2 ≤ k → ¬ (k ^ 2 ∣ a))
: (answer = a + b)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
