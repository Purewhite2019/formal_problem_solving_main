import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
Find all positive integers $n$ that are within $250$ of exactly $15$ perfect squares.

## Answer
the set of integers $n$ such that $315 \leq n \leq 325$ or $332 \leq n \leq 350$
-/

-- # Formal Answer
noncomputable abbrev Q155.gt := fun (n : ℤ) (answer : Set ℤ) => answer = ({n : ℤ | (315 : ℤ) ≤ n ∧ n ≤ (325 : ℤ) ∨ (332 : ℤ) ≤ n ∧ n ≤ (350 : ℤ)} : Set ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q155
  (n : ℤ)
find (answer : Set ℤ) s.t.
: (n ∈ answer ↔ (0 < n ∧ {m : ℕ | |n - m ^ 2| ≤ 250}.encard = 15))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q155
  (n : ℤ)
find_all (answer : Set ℤ) iff
: (n ∈ answer ↔ (0 < n ∧ {m : ℕ | |n - m ^ 2| ≤ 250}.encard = 15))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
