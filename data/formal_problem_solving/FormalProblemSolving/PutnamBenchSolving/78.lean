import FormalProblemSolving.Basic
open Set Function Filter Topology Polynomial Real

/-
## Problem
Let $p_n$ denote the probability that $c + d$ will be a perfect square if $c$ and $d$ are selected independently and uniformly at random from $\{1, 2, 3, \dots, n\}$. Express $\lim_{n \rightarrow \infty} p_n \sqrt{n}$ in the form $r(\sqrt{s} - t)$ for integers $s$ and $t$ and rational $r$.

## Answer
4/3 * (Real.sqrt 2 - 1)
-/

-- # Formal Answer
noncomputable abbrev Q78.gt := fun (answer : ℝ) => answer = ((4 / 3 : ℝ) * ((√(2 : ℝ) : ℝ) - (1 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q78
find (answer : ℝ) s.t.
  (p : ℕ → ℝ)
  (hp : p = fun n : ℕ => ({c : Finset.Icc 1 n × Finset.Icc 1 n | ∃ m : ℕ, m^2 = c.1 + c.2}.ncard : ℝ) / n^2)
: (Tendsto (fun n : ℕ => p n * Real.sqrt n) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q78
find_all (answer : ℝ) iff
  (p : ℕ → ℝ)
  (hp : p = fun n : ℕ => ({c : Finset.Icc 1 n × Finset.Icc 1 n | ∃ m : ℕ, m^2 = c.1 + c.2}.ncard : ℝ) / n^2)
: (Tendsto (fun n : ℕ => p n * Real.sqrt n) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
