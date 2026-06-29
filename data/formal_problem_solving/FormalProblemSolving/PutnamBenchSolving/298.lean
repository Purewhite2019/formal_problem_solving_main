import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
For every positive real number $x$, let $g(x)=\lim_{r \to 0}((x+1)^{r+1}-x^{r+1})^\frac{1}{r}$. Find $\lim_{x \to \infty}\frac{g(x)}{x}$.

## Answer
Real.exp 1
-/

-- # Formal Answer
noncomputable abbrev Q298.gt := fun (answer : ℝ) => answer = (Real.exp (1 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q298
find (answer : ℝ) s.t.
  (g : ℝ → ℝ)
  (hg : ∀ x > 0, Tendsto (fun r : ℝ => ((x + 1) ^ (r + 1) - x ^ (r + 1)) ^ (1 / r)) (𝓝[>] 0) (𝓝 (g x)))
: (Tendsto (fun x : ℝ => g x / x) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q298
find_all (answer : ℝ) iff
  (g : ℝ → ℝ)
  (hg : ∀ x > 0, Tendsto (fun r : ℝ => ((x + 1) ^ (r + 1) - x ^ (r + 1)) ^ (1 / r)) (𝓝[>] 0) (𝓝 (g x)))
: (Tendsto (fun x : ℝ => g x / x) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
