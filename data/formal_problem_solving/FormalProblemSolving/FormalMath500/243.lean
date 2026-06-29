import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose $a$ and $b$ are different prime numbers greater than 2. How many whole-number divisors are there for the integer $a(2a+b)-2a^{2}+ab$?

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q243.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q243
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (h₀ : Nat.Prime a)
  (h₁ : Nat.Prime b)
  (h₂ : a ≠ b)
  (h₃ : a > 2)
  (h₄ : b > 2)
: (answer = (Nat.divisors (a * (2 * a + b) - 2 * a^2 + a * b)).card)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q243
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (h₀ : Nat.Prime a)
  (h₁ : Nat.Prime b)
  (h₂ : a ≠ b)
  (h₃ : a > 2)
  (h₄ : b > 2)
: (answer = (Nat.divisors (a * (2 * a + b) - 2 * a^2 + a * b)).card)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
