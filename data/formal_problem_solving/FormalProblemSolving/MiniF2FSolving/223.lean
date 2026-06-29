import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A positive integer $n$ is nice if there is a positive integer $m$ with exactly four positive divisors (including $1$ and $m$) such that the sum of the four divisors is equal to $n.$ Find the sum of all the nice numbers in the set $\{ 2010,2011,2012,\dots,2019 \}.$

## Answer
2016
-/

-- # Formal Answer
noncomputable abbrev Q223.gt := fun (answer : ℕ) => answer = (2016 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q223
find (answer : ℕ) s.t.
  (S : Finset ℕ)
  (hS : S = {n | 2010 ≤ n ∧ n ≤ 2019 ∧ ∃ m, ((Nat.divisors m).card = 4 ∧ ∑ p in (Nat.divisors m), p = n)})
: (answer = ∑ k in S, k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q223
find_all (answer : ℕ) iff
  (S : Finset ℕ)
  (hS : S = {n | 2010 ≤ n ∧ n ≤ 2019 ∧ ∃ m, ((Nat.divisors m).card = 4 ∧ ∑ p in (Nat.divisors m), p = n)})
: (answer = ∑ k in S, k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
