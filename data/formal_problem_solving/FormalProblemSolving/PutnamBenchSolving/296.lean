import FormalProblemSolving.Basic
open Filter Topology Set

/-
## Problem
Let $n$ be a positive integer, and let $V_n$ be the set of integer $(2n+1)$-tuples $\mathbf{v} = (s_0, s_1, \cdots, s_{2n-1}, s_{2n})$ for which $s_0 = s_{2n} = 0$ and $|s_j - s_{j-1}| = 1$ for $j=1,2,\cdots,2n$. Define \[ q(\mathbf{v}) = 1 + \sum_{j=1}^{2n-1} 3^{s_j}, \] and let $M(n)$ be the average of $\frac{1}{q(\mathbf{v})}$ over all $\mathbf{v} \in V_n$. Evaluate $M(2020)$.

## Answer
1 / 4040
-/

-- # Formal Answer
noncomputable abbrev Q296.gt := fun (answer : ℝ) => answer = (1 / 4040 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q296
find (answer : ℝ) s.t.
  (V : ℕ → Set (ℕ → ℤ))
  (q : ℕ → (ℕ → ℤ) → ℝ)
  (M : ℕ → ℝ)
  (hV : V = fun n ↦ ({s : ℕ → ℤ | s 0 = 0 ∧ (∀ j ≥ 2 * n, s j = 0) ∧ (∀ j ∈ Icc 1 (2 * n), |s j - s (j - 1)| = 1)}))
  (hq : q = fun n s ↦ 1 + ∑ j in Finset.Icc 1 (2 * n - 1), 3 ^ (s j))
  (hM : M = fun n ↦ (∑' v : V n, 1 / (q n v)) / (V n).ncard)
: (M 2020 = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q296
find_all (answer : ℝ) iff
  (V : ℕ → Set (ℕ → ℤ))
  (q : ℕ → (ℕ → ℤ) → ℝ)
  (M : ℕ → ℝ)
  (hV : V = fun n ↦ ({s : ℕ → ℤ | s 0 = 0 ∧ (∀ j ≥ 2 * n, s j = 0) ∧ (∀ j ∈ Icc 1 (2 * n), |s j - s (j - 1)| = 1)}))
  (hq : q = fun n s ↦ 1 + ∑ j in Finset.Icc 1 (2 * n - 1), 3 ^ (s j))
  (hM : M = fun n ↦ (∑' v : V n, 1 / (q n v)) / (V n).ncard)
: (M 2020 = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
