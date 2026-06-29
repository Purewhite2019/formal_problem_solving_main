import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
For each positive integer $k$, let $A(k)$ be the number of odd divisors of $k$ in the interval $[1,\sqrt{2k})$. Evaluate $\sum_{k=1}^\infty (-1)^{k-1}\frac{A(k)}{k}$.

## Answer
π² / 16
-/

-- # Formal Answer
noncomputable abbrev Q262.gt := fun (answer : ℝ) => answer = (((Real.pi : ℝ) ^ (2 : ℕ) : ℝ) / (16 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q262
find (answer : ℝ) s.t.
  (A : ℕ → ℕ)
  (hA : ∀ k > 0, A k = {j : ℕ | Odd j ∧ j ∣ k ∧ j < Real.sqrt (2 * k)}.encard)
: (Tendsto (fun K : ℕ ↦ ∑ k in Finset.Icc 1 K, (-1 : ℝ) ^ ((k : ℝ) - 1) * (A k / (k : ℝ))) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q262
find_all (answer : ℝ) iff
  (A : ℕ → ℕ)
  (hA : ∀ k > 0, A k = {j : ℕ | Odd j ∧ j ∣ k ∧ j < Real.sqrt (2 * k)}.encard)
: (Tendsto (fun K : ℕ ↦ ∑ k in Finset.Icc 1 K, (-1 : ℝ) ^ ((k : ℝ) - 1) * (A k / (k : ℝ))) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
