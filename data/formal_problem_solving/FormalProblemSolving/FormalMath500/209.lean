import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sum of the digits of a two-digit number is $13.$ The difference between the number and the number with its digits reversed is $27.$ What is the sum of the original number and the number with its digits reversed?

## Answer
143
-/

-- # Formal Answer
noncomputable abbrev Q209.gt := fun (answer : ℝ) => answer = (143 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q209
find (answer : ℝ) s.t.
  (n : ℕ)
  (h₀ : n ≥ 10 ∧ n ≤ 99)
  (h₁ : n / 10 + (n % 10) = 13)
  (h₂ : n - ((n % 10) * 10 + n / 10) = 27)
: (n + (n % 10 * 10 + n / 10) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q209
find_all (answer : ℝ) iff
  (n : ℕ)
  (h₀ : n ≥ 10 ∧ n ≤ 99)
  (h₁ : n / 10 + (n % 10) = 13)
  (h₂ : n - ((n % 10) * 10 + n / 10) = 27)
: (n + (n % 10 * 10 + n / 10) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
