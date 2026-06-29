import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $n$ be a positive integer.  What is the greatest possible value of $\gcd(n + 7, 2n + 1)$?

## Answer
13
-/

-- # Formal Answer
noncomputable abbrev Q128.gt := fun (answer : ℕ) => answer = (13 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q128
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (n : ℕ)
  (h₀ : 0 < n)
  (hS : S = {d | ∃ n, 0 < n ∧ d = Nat.gcd (n + 7) (2 * n + 1)})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q128
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (n : ℕ)
  (h₀ : 0 < n)
  (hS : S = {d | ∃ n, 0 < n ∧ d = Nat.gcd (n + 7) (2 * n + 1)})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
