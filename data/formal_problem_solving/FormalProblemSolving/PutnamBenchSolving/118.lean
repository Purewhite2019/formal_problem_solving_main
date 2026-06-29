import FormalProblemSolving.Basic
open Set Filter Topology

/-
## Problem
Determine, with proof, the set of real numbers $x$ for which
\[
\sum_{n=1}^\infty \left( \frac{1}{n} \csc \frac{1}{n} - 1 \right)^x
\]
converges.

## Answer
{x | x > 1 / 2}
-/

-- # Formal Answer
noncomputable abbrev Q118.gt := fun (answer : Set ℝ) => answer = ({x : ℝ | x > (1 / 2 : ℝ)} : Set ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q118
find (answer : Set ℝ) s.t.
: (answer = {x : ℝ | ∃ L : ℝ, Tendsto (fun t ↦ ∑ n in Finset.Icc (1 : ℕ) t, (((1 / n) / Real.sin (1 / n) - 1) ^ x)) atTop (𝓝 L)})
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q118
find_all (answer : Set ℝ) iff
: (answer = {x : ℝ | ∃ L : ℝ, Tendsto (fun t ↦ ∑ n in Finset.Icc (1 : ℕ) t, (((1 / n) / Real.sin (1 / n) - 1) ^ x)) atTop (𝓝 L)})
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
