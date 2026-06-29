import FormalProblemSolving.Basic
open Filter Topology Bornology Set

/-
## Problem
For each positive integer $n$, write the sum $\sum_{m=1}^n 1/m$ in the form $p_n/q_n$, where $p_n$ and $q_n$ are relatively prime positive integers.  Determine all $n$ such that 5 does not divide $q_n$.

## Answer
{n | (1 ≤ n ∧ n ≤ 4) ∨ (20 ≤ n ∧ n ≤ 24) ∨ (100 ≤ n ∧ n ≤ 104) ∨ (120 ≤ n ∧ n ≤ 124)}
-/

-- # Formal Answer
noncomputable abbrev Q173.gt := fun (n : ℕ) (answer : Set ℕ) => answer =
  ({n : ℕ |
      (1 : ℕ) ≤ n ∧ n ≤ (4 : ℕ) ∨
        (20 : ℕ) ≤ n ∧ n ≤ (24 : ℕ) ∨ (100 : ℕ) ≤ n ∧ n ≤ (104 : ℕ) ∨ (120 : ℕ) ≤ n ∧ n ≤ (124 : ℕ)} :
    Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q173
  (n : ℕ)
find (answer : Set ℕ) s.t.
  (hn : n > 0)
: (n ∈ answer ↔ ¬5 ∣ (∑ m in Finset.Icc 1 n, 1/m : ℚ).den)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q173
  (n : ℕ)
find_all (answer : Set ℕ) iff
  (hn : n > 0)
: (n ∈ answer ↔ ¬5 ∣ (∑ m in Finset.Icc 1 n, 1/m : ℚ).den)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
