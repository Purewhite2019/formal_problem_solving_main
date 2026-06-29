import FormalProblemSolving.Basic
open Set

/-
## Problem
For each continuous function $f: [0,1] \to \mathbb{R}$, let $I(f) = \int_0^1 x^2 f(x)\,dx$ and $J(x) = \int_0^1 x \left(f(x)\right)^2\,dx$. Find the maximum value of $I(f) - J(f)$ over all such functions $f$.

## Answer
$\frac{1}{16}$
-/

-- # Formal Answer
noncomputable abbrev Q208.gt := fun (answer : ℝ) => answer = (1 / 16 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q208
find (answer : ℝ) s.t.
  (I : (ℝ → ℝ) → ℝ)
  (J : (ℝ → ℝ) → ℝ)
  (hI : I = fun f ↦ ∫ x in (0)..1, x ^ 2 * (f x))
  (hJ : J = fun f ↦ ∫ x in (0)..1, x * (f x) ^ 2)
: (IsGreatest {y | ∃ f : ℝ → ℝ, ContinuousOn f (Icc 0 1) ∧ I f - J f = y} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q208
find_all (answer : ℝ) iff
  (I : (ℝ → ℝ) → ℝ)
  (J : (ℝ → ℝ) → ℝ)
  (hI : I = fun f ↦ ∫ x in (0)..1, x ^ 2 * (f x))
  (hJ : J = fun f ↦ ∫ x in (0)..1, x * (f x) ^ 2)
: (IsGreatest {y | ∃ f : ℝ → ℝ, ContinuousOn f (Icc 0 1) ∧ I f - J f = y} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
