import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose $a$ and $b$ are positive integers such that the units digit of $a$ is $2$, the units digit of $b$ is $4$, and the greatest common divisor of $a$ and $b$ is $6$.

What is the smallest possible value of the least common multiple of $a$ and $b$?

## Answer
108
-/

-- # Formal Answer
noncomputable abbrev Q249.gt := fun (answer : ℕ) => answer = (108 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q249
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (a : ℕ)
  (b : ℕ)
  (h₀ : 0 < a ∧ 0 < b)
  (h₁ : a % 10 = 2)
  (h₂ : b % 10 = 4)
  (h₃ : Nat.gcd a b = 6)
  (hS : S = {l : ℕ | ∃ a b, 0 < a ∧ 0 < b ∧ a % 10 = 2 ∧ b % 10 = 4 ∧ Nat.gcd a b = 6 ∧ l = Nat.lcm a b})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q249
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (a : ℕ)
  (b : ℕ)
  (h₀ : 0 < a ∧ 0 < b)
  (h₁ : a % 10 = 2)
  (h₂ : b % 10 = 4)
  (h₃ : Nat.gcd a b = 6)
  (hS : S = {l : ℕ | ∃ a b, 0 < a ∧ 0 < b ∧ a % 10 = 2 ∧ b % 10 = 4 ∧ Nat.gcd a b = 6 ∧ l = Nat.lcm a b})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
