import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $r$ be the remainder when $1342$ is divided by $13$.

Determine the smallest positive integer that has these two properties:

$\bullet~$ It is a multiple of $1342$.

$\bullet~$ Its remainder upon being divided by $13$ is smaller than $r$.

## Answer
6710
-/

-- # Formal Answer
noncomputable abbrev Q256.gt := fun (answer : ℕ) => answer = (6710 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q256
find (answer : ℕ) s.t.
  (r : ℕ)
  (h₀ : r = 1342 % 13)
  (S : Set ℕ)
  (hS : S = {n | 0 < n ∧ 1342 ∣ n ∧ n % 13 < r})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q256
find_all (answer : ℕ) iff
  (r : ℕ)
  (h₀ : r = 1342 % 13)
  (S : Set ℕ)
  (hS : S = {n | 0 < n ∧ 1342 ∣ n ∧ n % 13 < r})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
