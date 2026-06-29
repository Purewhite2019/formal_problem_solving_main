import FormalProblemSolving.Basic
open Set

/-
## Problem
Call a set of positive integers 'conspiratorial' if no three of them are pairwise relatively prime. What is the largest number of elements in any conspiratorial subset of the integers 1 through 16?

## Answer
11
-/

-- # Formal Answer
noncomputable abbrev Q28.gt := fun (answer : ℕ) => answer = (11 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q28
find (answer : ℕ) s.t.
  (conspiratorial : Set ℤ → Prop)
  (hconspiratorial : ∀ S, conspiratorial S ↔ ∀ a ∈ S, ∀ b ∈ S, ∀ c ∈ S, (a > 0 ∧ b > 0 ∧ c > 0) ∧ ((a ≠ b ∧ b ≠ c ∧ a ≠ c) → (Int.gcd a b > 1 ∨ Int.gcd b c > 1 ∨ Int.gcd a c > 1)))
: (IsGreatest {k | ∃ S, S ⊆ Icc 1 16 ∧ conspiratorial S ∧ S.encard = k} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q28
find_all (answer : ℕ) iff
  (conspiratorial : Set ℤ → Prop)
  (hconspiratorial : ∀ S, conspiratorial S ↔ ∀ a ∈ S, ∀ b ∈ S, ∀ c ∈ S, (a > 0 ∧ b > 0 ∧ c > 0) ∧ ((a ≠ b ∧ b ≠ c ∧ a ≠ c) → (Int.gcd a b > 1 ∨ Int.gcd b c > 1 ∨ Int.gcd a c > 1)))
: (IsGreatest {k | ∃ S, S ⊆ Icc 1 16 ∧ conspiratorial S ∧ S.encard = k} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
