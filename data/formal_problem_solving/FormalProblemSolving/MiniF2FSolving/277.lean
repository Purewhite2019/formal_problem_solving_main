import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Euler discovered that the polynomial $p(n) = n^2 - n + 41$ yields prime numbers for many small positive integer values of $n$. What is the smallest positive integer $n$ for which $p(n)$ and $p(n+1)$ share a common factor greater than $1$?

## Answer
41
-/

-- # Formal Answer
noncomputable abbrev Q277.gt := fun (answer : ℕ) => answer = (41 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q277
find (answer : ℕ) s.t.
  (p : ℕ → ℕ)
  (h₀ : ∀ x, p x = x ^ 2 - x + 41)
: (IsLeast {n : ℕ | 0 < n ∧ 1 < Nat.gcd (p n) (p (n + 1))} answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q277
find_all (answer : ℕ) iff
  (p : ℕ → ℕ)
  (h₀ : ∀ x, p x = x ^ 2 - x + 41)
: (IsLeast {n : ℕ | 0 < n ∧ 1 < Nat.gcd (p n) (p (n + 1))} answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
