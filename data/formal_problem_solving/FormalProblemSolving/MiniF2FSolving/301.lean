import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the sum of the final three digits of the integer representation of $5^{100}$?

## Answer
13
-/

-- # Formal Answer
noncomputable abbrev Q301.gt := fun (answer : ℕ) => answer = (13 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q301
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (h₀ : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (h₁ : Nat.digits 10 ((5^100) % 1000) = [c,b,a])
: (answer = a + b + c)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q301
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (h₀ : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (h₁ : Nat.digits 10 ((5^100) % 1000) = [c,b,a])
: (answer = a + b + c)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
