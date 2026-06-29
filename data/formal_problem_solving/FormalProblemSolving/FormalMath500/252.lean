import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the smallest positive multiple of 450 whose digits are all zeroes and ones?

## Answer
11,\! 111,\! 111,\! 100
-/

-- # Formal Answer
noncomputable abbrev Q252.gt := fun (answer : ℕ) => answer = (11111111100 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q252
find (answer : ℕ) s.t.
  (n : ℕ)
: (∀ (n : ℕ), n > 0 ∧ n % 450 = 0 ∧ (∀ d ∈ Nat.digits 10 n, d = 0 ∨ d = 1) → n ≥ answer) ∧ (∃ (n : ℕ), n > 0 ∧ n % 450 = 0 ∧ (∀ d ∈ Nat.digits 10 n, d = 0 ∨ d = 1) ∧ answer = n)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q252
find_all (answer : ℕ) iff
  (n : ℕ)
: (∀ (n : ℕ), n > 0 ∧ n % 450 = 0 ∧ (∀ d ∈ Nat.digits 10 n, d = 0 ∨ d = 1) → n ≥ answer) ∧ (∃ (n : ℕ), n > 0 ∧ n % 450 = 0 ∧ (∀ d ∈ Nat.digits 10 n, d = 0 ∨ d = 1) ∧ answer = n)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
