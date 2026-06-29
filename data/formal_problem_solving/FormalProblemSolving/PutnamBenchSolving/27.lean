import FormalProblemSolving.Basic
open Nat Set MeasureTheory Topology Filter

/-
## Problem
Suppose $f$ is a function on $[0,1]$ with continuous derivative satisfying $0 < f'(x) \leq 1$ and $f 0 = 0$. Prove that $\left[\int_0^1 f(x) dx\right]]^2 \geq \int_0^1 (f(x))^3 dx$, and find an example where equality holds.

## Answer
the identity function $f(x) = x$
-/

-- # Formal Answer
noncomputable abbrev Q27.gt := fun (answer : ℝ → ℝ) => answer = fun (x : ℝ) => x

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q27
find (answer : ℝ → ℝ) s.t.
: (ContDiff ℝ 1 answer ∧ (∀ x : ℝ, 0 < deriv answer x ∧ deriv answer x ≤ 1) ∧ answer 0 = 0 ∧ (∫ x in Icc 0 1, answer x)^2 = ∫ x in Icc 0 1, (answer x)^3)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q27
find_all (answer : ℝ → ℝ) iff
: (ContDiff ℝ 1 answer ∧ (∀ x : ℝ, 0 < deriv answer x ∧ deriv answer x ≤ 1) ∧ answer 0 = 0 ∧ (∫ x in Icc 0 1, answer x)^2 = ∫ x in Icc 0 1, (answer x)^3)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
