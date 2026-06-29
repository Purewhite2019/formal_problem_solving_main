import FormalProblemSolving.Basic
open Metric Set EuclideanGeometry Filter Topology

/-
## Problem
Evaluate the infinite product $\lim_{n \to \infty} \frac{1}{n^4} \prod_{i = 1}^{2n} (n^2 + i^2)^{1/n}$.

## Answer
$e^{2 \ln 5 - 4 + 2 \arctan 2}$
-/

-- # Formal Answer
noncomputable abbrev Q19.gt := fun (answer : ℝ) => answer =
  (Real.exp ((((2 : ℝ) * (Real.log (5 : ℝ) : ℝ) : ℝ) - (4 : ℝ) : ℝ) + ((2 : ℝ) * (Real.arctan (2 : ℝ) : ℝ) : ℝ) : ℝ) :
    ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q19
find (answer : ℝ) s.t.
: (Tendsto (fun n => 1/(n^4) * ∏ i in Finset.Icc (1 : ℤ) (2*n), ((n^2 + i^2) : ℝ)^((1 : ℝ)/n)) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q19
find_all (answer : ℝ) iff
: (Tendsto (fun n => 1/(n^4) * ∏ i in Finset.Icc (1 : ℤ) (2*n), ((n^2 + i^2) : ℝ)^((1 : ℝ)/n)) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
