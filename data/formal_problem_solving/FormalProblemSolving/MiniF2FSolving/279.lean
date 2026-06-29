import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the smallest positive integer $k$ such that, for every positive integer $n$, $6n+k$ is relatively prime to each of $6n+3$, $6n+2$, and $6n+1$.

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q279.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q279
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {k : ℕ | ∀ n, Nat.gcd (6 * n + k) (6 * n + 3) = 1 ∧
                      Nat.gcd (6 * n + k) (6 * n + 2) = 1 ∧
                      Nat.gcd (6 * n + k) (6 * n + 1) = 1})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q279
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {k : ℕ | ∀ n, Nat.gcd (6 * n + k) (6 * n + 3) = 1 ∧
                      Nat.gcd (6 * n + k) (6 * n + 2) = 1 ∧
                      Nat.gcd (6 * n + k) (6 * n + 1) = 1})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
