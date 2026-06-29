import FormalProblemSolving.Basic
open RingHom Set Nat Filter Topology

/-
## Problem
Find $\prod_{n=2}^{\infty} \frac{(n^3 - 1)}{(n^3 + 1)}$.

## Answer
2/3
-/

-- # Formal Answer
noncomputable abbrev Q50.gt := fun (answer : ℝ) => answer = (2 / 3 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q50
find (answer : ℝ) s.t.
: (Tendsto (fun N ↦ ∏ n in Finset.Icc (2 : ℤ) N, ((n : ℝ) ^ 3 - 1) / (n ^ 3 + 1)) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q50
find_all (answer : ℝ) iff
: (Tendsto (fun N ↦ ∏ n in Finset.Icc (2 : ℤ) N, ((n : ℝ) ^ 3 - 1) / (n ^ 3 + 1)) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
