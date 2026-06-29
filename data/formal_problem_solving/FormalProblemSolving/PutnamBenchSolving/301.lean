import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
Let $A$ be the set of all integers $n$ such that $1 \leq n \leq 2021$ and $\gcd(n,2021)=1$. For every nonnegative integer $j$, let $S(j)=\sum_{n \in A}n^j$. Determine all values of $j$ such that $S(j)$ is a multiple of $2021$.

## Answer
the set of all j such that neither 42 nor 46 divides j
-/

-- # Formal Answer
noncomputable abbrev Q301.gt := fun (j : ℕ) (answer : Set ℕ) => answer = ({j : ℕ | ¬(42 : ℕ) ∣ j ∧ ¬(46 : ℕ) ∣ j} : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q301
  (j : ℕ)
find (answer : Set ℕ) s.t.
  (A : Finset ℕ)
  (S : ℕ → ℕ)
  (hA : A = {n | 1 ≤ n ∧ n ≤ 2021 ∧ Nat.gcd n 2021 = 1})
  (hS : ∀ j' : ℕ, S j' = ∑ n in A, n ^ j')
: ((2021 ∣ S j) ↔ j ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q301
  (j : ℕ)
find_all (answer : Set ℕ) iff
  (A : Finset ℕ)
  (S : ℕ → ℕ)
  (hA : A = {n | 1 ≤ n ∧ n ≤ 2021 ∧ Nat.gcd n 2021 = 1})
  (hS : ∀ j' : ℕ, S j' = ∑ n in A, n ^ j')
: ((2021 ∣ S j) ↔ j ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
