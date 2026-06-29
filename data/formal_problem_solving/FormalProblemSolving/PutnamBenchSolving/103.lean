import FormalProblemSolving.Basic
open Real

/-
## Problem
Evaluate $\sum_{n=0}^\infty \mathrm{Arccot}(n^2+n+1)$, where $\mathrm{Arccot}\,t$ for $t \geq 0$ denotes the number $\theta$ in the interval $0 < \theta \leq \pi/2$ with $\cot \theta = t$.

## Answer
π / 2
-/

-- # Formal Answer
noncomputable abbrev Q103.gt := fun (answer : ℝ) => answer = ((π : ℝ) / (2 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q103
find (answer : ℝ) s.t.
  (cot : ℝ → ℝ)
  (fcot : cot = fun θ ↦ Real.cos θ / Real.sin θ)
  (arccot : ℝ → ℝ)
  (harccot : ∀ t : ℝ, t ≥ 0 → arccot t ∈ Set.Ioc 0 (Real.pi / 2) ∧ cot (arccot t) = t)
: ((∑' n : ℕ, arccot (n ^ 2 + n + 1)) = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q103
find_all (answer : ℝ) iff
  (cot : ℝ → ℝ)
  (fcot : cot = fun θ ↦ Real.cos θ / Real.sin θ)
  (arccot : ℝ → ℝ)
  (harccot : ∀ t : ℝ, t ≥ 0 → arccot t ∈ Set.Ioc 0 (Real.pi / 2) ∧ cot (arccot t) = t)
: ((∑' n : ℕ, arccot (n ^ 2 + n + 1)) = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
