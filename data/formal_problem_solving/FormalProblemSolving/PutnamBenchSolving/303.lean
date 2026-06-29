import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
Determine the maximum value of the sum $S = \sum_{n=1}^\infty \frac{n}{2^n}(a_1a_2 \dots a_n)^{1/n}$ over all sequences $a_1,a_2,a_3,\dots$ of nonnegative real numbers satisfying $\sum_{k=1}^\infty a_k=1$.

## Answer
2 / 3
-/

-- # Formal Answer
noncomputable abbrev Q303.gt := fun (answer : ℝ) => answer = (2 / 3 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q303
find (answer : ℝ) s.t.
: (IsGreatest
      {S | ∃ a : ℕ+ → ℝ, (∑' k, a k = 1) ∧ (∀ k, 0 ≤ a k) ∧
        S = ∑' n : ℕ+, n / 2 ^ (n : ℕ) * (∏ k in Finset.Icc 1 n, a k) ^ (1 / n : ℝ)}
      answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q303
find_all (answer : ℝ) iff
: (IsGreatest
      {S | ∃ a : ℕ+ → ℝ, (∑' k, a k = 1) ∧ (∀ k, 0 ≤ a k) ∧
        S = ∑' n : ℕ+, n / 2 ^ (n : ℕ) * (∏ k in Finset.Icc 1 n, a k) ^ (1 / n : ℝ)}
      answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
