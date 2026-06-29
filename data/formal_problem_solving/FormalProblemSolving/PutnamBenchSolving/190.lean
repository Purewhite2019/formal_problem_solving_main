import FormalProblemSolving.Basic
open MvPolynomial

/-
## Problem
Let $n$ be a fixed positive integer. How many ways are there to write $n$ as a sum of positive integers, \[ n = a_1 + a_2 + \dots + a_k, \] with $k$ an arbitrary positive integer and $a_1 \leq a_2 \leq \dots \leq a_k \leq a_1 + 1$? For example, with $n = 4$, there are four ways: $4, 2 + 2, 1 + 1 + 2, 1 + 1 + 1 + 1$

## Answer
$n$
-/

-- # Formal Answer
noncomputable abbrev Q190.gt := fun (n : ℕ) (answer : ℕ → ℕ) => answer = fun (n : ℕ) => n

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q190
  (n : ℕ)
find (answer : ℕ → ℕ) s.t.
  (hn : n > 0)
: (Set.encard {a : ℕ → ℤ |
  ∃ k > 0, (∑ i : Fin k, a i = n) ∧
    (∀ i : Fin k, a i > 0) ∧
    (∀ i : Fin (k - 1), a i ≤ a (i + 1)) ∧
    a (k - 1) ≤ a 0 + 1 ∧ (∀ i ≥ k, a i = 0)} = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q190
  (n : ℕ)
find_all (answer : ℕ → ℕ) iff
  (hn : n > 0)
: (Set.encard {a : ℕ → ℤ |
  ∃ k > 0, (∑ i : Fin k, a i = n) ∧
    (∀ i : Fin k, a i > 0) ∧
    (∀ i : Fin (k - 1), a i ≤ a (i + 1)) ∧
    a (k - 1) ≤ a 0 + 1 ∧ (∀ i ≥ k, a i = 0)} = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
