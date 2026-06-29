import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
Find the set of all real numbers $k$ with the following property: For any positive, differentiable function $f$ that satisfies $f'(x)>f(x)$ for all $x$, there is some number $N$ such that $f(x)>e^{kx}$ for all $x>N$.

## Answer
the set of all real numbers less than 1
-/

-- # Formal Answer
noncomputable abbrev Q157.gt := fun (answer : Set ℝ) => answer = (Set.Iio (1 : ℝ) : Set ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q157
find (answer : Set ℝ) s.t.
: ({k | ∀ f (hf : (∀ x, 0 < f x ∧ f x < deriv f x) ∧ Differentiable ℝ f),
      ∃ N, ∀ x > N, Real.exp (k * x) < f x} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q157
find_all (answer : Set ℝ) iff
: ({k | ∀ f (hf : (∀ x, 0 < f x ∧ f x < deriv f x) ∧ Differentiable ℝ f),
      ∃ N, ∀ x > N, Real.exp (k * x) < f x} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
