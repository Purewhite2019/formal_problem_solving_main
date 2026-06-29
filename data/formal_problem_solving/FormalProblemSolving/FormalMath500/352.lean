import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $(a_1, a_2, \dots, a_n)$ be a sequence of positive real numbers, such that
\[\sum_{i = 1}^n a_i = 96, \quad \sum_{i = 1}^n a_i^2 = 144, \quad \sum_{i = 1}^n a_i^3 = 216.\]Find the sum of all possible values of $n.$

## Answer
64
-/

-- # Formal Answer
noncomputable abbrev Q352.gt := fun (answer : ℕ) => answer = (64 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q352
find (answer : ℕ) s.t.
  (SumSet : Finset ℕ)
  (hSumSet : ∀ n, n ∈ SumSet ↔ ∃ (a : ℕ → ℝ), (∀ i, 1 ≤ i ∧ i ≤ n → a i > 0) ∧
    (∑ i in Finset.range n, a (i + 1) = 96) ∧
    (∑ i in Finset.range n, (a (i + 1)) ^ 2 = 144) ∧
    (∑ i in Finset.range n, (a (i + 1)) ^ 3 = 216))
: (answer = ∑ n in SumSet, n)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q352
find_all (answer : ℕ) iff
  (SumSet : Finset ℕ)
  (hSumSet : ∀ n, n ∈ SumSet ↔ ∃ (a : ℕ → ℝ), (∀ i, 1 ≤ i ∧ i ≤ n → a i > 0) ∧
    (∑ i in Finset.range n, a (i + 1) = 96) ∧
    (∑ i in Finset.range n, (a (i + 1)) ^ 2 = 144) ∧
    (∑ i in Finset.range n, (a (i + 1)) ^ 3 = 216))
: (answer = ∑ n in SumSet, n)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
