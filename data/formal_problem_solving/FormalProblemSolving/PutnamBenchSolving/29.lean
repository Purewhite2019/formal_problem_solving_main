import FormalProblemSolving.Basic
open Set

/-
## Problem
A well-known theorem asserts that a prime $p > 2$ can be written as the sum of two perfect squres if and only if $p \equiv 1 \bmod 4$. Find which primes $p > 2$ can be written in each of the following forms, using (not necessarily positive) integers $x$ and $y$: (a) $x^2 + 16y^2$, (b) $4x^2 + 4xy + 5y^2$.

## Answer
({p : ℕ | p.Prime ∧ p ≡ 1 [MOD 8]}, {p : ℕ | p.Prime ∧ p ≡ 5 [MOD 8]})
-/

-- # Formal Answer
noncomputable abbrev Q29.gt := fun (p : ℕ) (answer : (Set ℕ) × (Set ℕ)) => answer =
  ((({p : ℕ | Nat.Prime p ∧ p ≡ (1 : ℕ) [MOD (8 : ℕ)]} : Set ℕ),
      ({p : ℕ | Nat.Prime p ∧ p ≡ (5 : ℕ) [MOD (8 : ℕ)]} : Set ℕ)) :
    Set ℕ × Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q29
  (p : ℕ)
find (answer : (Set ℕ) × (Set ℕ)) s.t.
  (h_assumption : ∀ p : ℕ, p.Prime ∧ p > 2 → ((∃ m n : ℤ, p = m^2 + n^2) ↔ p ≡ 1 [MOD 4]))
: (∀ p : ℕ,
    ((p.Prime ∧ p > 2 ∧ (∃ x y : ℤ, p = x^2 + 16*y^2)) ↔ p ∈ answer.1) ∧
    ((p.Prime ∧ p > 2 ∧ (∃ x y : ℤ, p = 4*x^2 + 4*x*y + 5*y^2)) ↔ p ∈ answer.2))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q29
  (p : ℕ)
find_all (answer : (Set ℕ) × (Set ℕ)) iff
  (h_assumption : ∀ p : ℕ, p.Prime ∧ p > 2 → ((∃ m n : ℤ, p = m^2 + n^2) ↔ p ≡ 1 [MOD 4]))
: (∀ p : ℕ,
    ((p.Prime ∧ p > 2 ∧ (∃ x y : ℤ, p = x^2 + 16*y^2)) ↔ p ∈ answer.1) ∧
    ((p.Prime ∧ p > 2 ∧ (∃ x y : ℤ, p = 4*x^2 + 4*x*y + 5*y^2)) ↔ p ∈ answer.2))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
