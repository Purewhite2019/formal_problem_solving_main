import FormalProblemSolving.Basic
open Set Filter Topology

/-
## Problem
For every $n$ in the set $N=\{1,2,\dots\}$ of positive integers, let $r_n$ be the minimum value of $|c-d \sqrt{3}|$ for all nonnegative integers $c$ and $d$ with $c+d=n$. Find, with proof, the smallest positive real number $g$ with $r_n \leq g$ for all $n \in N$.

## Answer
(1 + √3) / 2
-/

-- # Formal Answer
noncomputable abbrev Q122.gt := fun (answer : ℝ) => answer = (((1 : ℝ) + (√(3 : ℝ) : ℝ) : ℝ) / (2 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q122
find (answer : ℝ) s.t.
  (r : ℤ → ℝ)
  (hr : ∀ n ≥ 1,
      (∃ c d : ℤ,
        (c ≥ 0 ∧ d ≥ 0) ∧
        c + d = n ∧ r n = |c - d * Real.sqrt 3|) ∧
        (∀ c d : ℤ, (c ≥ 0 ∧ d ≥ 0 ∧ c + d = n) → |c - d * Real.sqrt 3| ≥ r n))
: (IsLeast {g : ℝ | g > 0 ∧ (∀ n : ℤ, n ≥ 1 → r n ≤ g)} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q122
find_all (answer : ℝ) iff
  (r : ℤ → ℝ)
  (hr : ∀ n ≥ 1,
      (∃ c d : ℤ,
        (c ≥ 0 ∧ d ≥ 0) ∧
        c + d = n ∧ r n = |c - d * Real.sqrt 3|) ∧
        (∀ c d : ℤ, (c ≥ 0 ∧ d ≥ 0 ∧ c + d = n) → |c - d * Real.sqrt 3| ≥ r n))
: (IsLeast {g : ℝ | g > 0 ∧ (∀ n : ℤ, n ≥ 1 → r n ≤ g)} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
