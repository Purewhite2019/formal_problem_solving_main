import FormalProblemSolving.Basic
open Filter Topology Set

/-
## Problem
For a positive integer $N$, let $f_N$ be the function defined by
\[
f_N(x) = \sum_{n=0}^N \frac{N+1/2-n}{(N+1)(2n+1)} \sin((2n+1)x).
\]
Determine the smallest constant $M$ such that $f_N(x) \leq M$ for all $N$ and all real $x$.

## Answer
π/4
-/

-- # Formal Answer
noncomputable abbrev Q294.gt := fun (answer : ℝ) => answer = ((Real.pi : ℝ) / (4 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q294
find (answer : ℝ) s.t.
  (f : ℤ → (ℝ → ℝ))
  (hf : f = fun N : ℤ => fun x : ℝ =>
    ∑ n in Finset.Icc 0 N, (N + 1/2 - n)/((N + 1)*(2*n + 1)) * Real.sin ((2*n + 1)*x))
: (answer = sSup {y | ∃ᵉ (N > 0) (x : ℝ), y = f N x})
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q294
find_all (answer : ℝ) iff
  (f : ℤ → (ℝ → ℝ))
  (hf : f = fun N : ℤ => fun x : ℝ =>
    ∑ n in Finset.Icc 0 N, (N + 1/2 - n)/((N + 1)*(2*n + 1)) * Real.sin ((2*n + 1)*x))
: (answer = sSup {y | ∃ᵉ (N > 0) (x : ℝ), y = f N x})
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
