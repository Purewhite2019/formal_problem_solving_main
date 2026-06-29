import FormalProblemSolving.Basic
open Nat Filter Topology Real

/-
## Problem
Define $\left\lVert x \right\rVert$ as the distance from $x$ to the nearest integer. Find $\lim_{n \to \infty} \frac{1}{n} \int_{1}^{n} \left\lVert \frac{n}{x} \right\rVert \, dx$. You may assume that $\prod_{n=1}^{\infty} \frac{2n}{(2n-1)} \cdot \frac{2n}{(2n+1)} = \frac{\pi}{2}$.

## Answer
$\log \left(\frac{4}{\pi}\right)$
-/

-- # Formal Answer
noncomputable abbrev Q84.gt := fun (answer : ℝ) => answer = (Real.log ((4 : ℝ) / (π : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q84
find (answer : ℝ) s.t.
  (dist_fun : ℝ → ℝ)
  (hdist_fun : dist_fun = fun (x : ℝ) ↦ min (x - ⌊x⌋) (⌈x⌉ - x))
  (fact : Tendsto (fun N ↦ ∏ n in Finset.Icc 1 N, (2 * n / (2 * n - 1)) * (2 * n / (2 * n + 1)) : ℕ → ℝ) atTop (𝓝 (Real.pi / 2)))
: (Tendsto (fun n ↦ (1 / n) * ∫ x in (1)..n, dist_fun (n / x) : ℕ → ℝ) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q84
find_all (answer : ℝ) iff
  (dist_fun : ℝ → ℝ)
  (hdist_fun : dist_fun = fun (x : ℝ) ↦ min (x - ⌊x⌋) (⌈x⌉ - x))
  (fact : Tendsto (fun N ↦ ∏ n in Finset.Icc 1 N, (2 * n / (2 * n - 1)) * (2 * n / (2 * n + 1)) : ℕ → ℝ) atTop (𝓝 (Real.pi / 2)))
: (Tendsto (fun n ↦ (1 / n) * ∫ x in (1)..n, dist_fun (n / x) : ℕ → ℝ) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
