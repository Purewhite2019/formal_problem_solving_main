import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The greatest common divisor of positive integers $m$ and $n$ is 8. The least common multiple of $m$ and $n$ is 112. What is the least possible value of $m+n$?

## Answer
72
-/

-- # Formal Answer
noncomputable abbrev Q302.gt := fun (answer : ℕ) => answer = (72 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q302
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (m : ℕ)
  (n : ℕ)
  (h₀ : 0 < m ∧ 0 < n)
  (h₁ : Nat.gcd m n = 8)
  (h₂ : Nat.lcm m n = 112)
  (hS : S = {k | ∃ m n, 0 < m ∧ 0 < n ∧ Nat.gcd m n = 8 ∧ Nat.lcm m n = 112 ∧ k = m + n})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q302
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (m : ℕ)
  (n : ℕ)
  (h₀ : 0 < m ∧ 0 < n)
  (h₁ : Nat.gcd m n = 8)
  (h₂ : Nat.lcm m n = 112)
  (hS : S = {k | ∃ m n, 0 < m ∧ 0 < n ∧ Nat.gcd m n = 8 ∧ Nat.lcm m n = 112 ∧ k = m + n})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
