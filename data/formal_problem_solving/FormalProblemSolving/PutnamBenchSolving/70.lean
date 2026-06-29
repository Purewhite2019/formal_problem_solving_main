import FormalProblemSolving.Basic
open Topology Filter Set Polynomial Function

/-
## Problem
Find the value of $$\lim_{n \rightarrow \infty} \frac{1}{n^5}\sum_{h=1}^{n}\sum_{k=1}^{n}(5h^4 - 18h^2k^2 + 5k^4).$$

## Answer
-1
-/

-- # Formal Answer
noncomputable abbrev Q70.gt := fun (answer : ℝ) => answer = (-1 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q70
find (answer : ℝ) s.t.
  (f : ℕ → ℝ)
  (hf : f = fun n : ℕ => ((1 : ℝ)/n^5) * ∑ h in Finset.Icc 1 n, ∑ k in Finset.Icc 1 n, (5*(h : ℝ)^4 - 18*h^2*k^2 + 5*k^4))
: (Tendsto f atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q70
find_all (answer : ℝ) iff
  (f : ℕ → ℝ)
  (hf : f = fun n : ℕ => ((1 : ℝ)/n^5) * ∑ h in Finset.Icc 1 n, ∑ k in Finset.Icc 1 n, (5*(h : ℝ)^4 - 18*h^2*k^2 + 5*k^4))
: (Tendsto f atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
