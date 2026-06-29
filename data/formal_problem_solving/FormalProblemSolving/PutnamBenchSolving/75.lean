import FormalProblemSolving.Basic
open Set Function Filter Topology Polynomial Real

/-
## Problem
Evaluate $\int_0^{\infty} \frac{\tan^{-1}(\pi x) - \tan^{-1} x}{x} \, dx$.

## Answer
$\frac{\pi}{2} \log \pi$
-/

-- # Formal Answer
noncomputable abbrev Q75.gt := fun (answer : ℝ) => answer = (((π : ℝ) / (2 : ℝ) : ℝ) * (Real.log (π : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q75
find (answer : ℝ) s.t.
: (Tendsto (fun t ↦ ∫ x in (0)..t, (arctan (Real.pi * x) - arctan x) / x) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q75
find_all (answer : ℝ) iff
: (Tendsto (fun t ↦ ∫ x in (0)..t, (arctan (Real.pi * x) - arctan x) / x) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
