import FormalProblemSolving.Basic

/-
## Problem
Find all differentiable functions $f:\mathbb{R} \to \mathbb{R}$ such that
\[
f'(x) = \frac{f(x+n)-f(x)}{n}
\]
for all real numbers $x$ and all positive integers $n$.

## Answer
{f : ℝ → ℝ | ∃ c d : ℝ, ∀ x : ℝ, f x = c*x + d}
-/

-- # Formal Answer
noncomputable abbrev Q230.gt := fun (answer : Set (ℝ → ℝ)) => answer = ({f : ℝ → ℝ | ∃ (c : ℝ) (d : ℝ), ∀ (x : ℝ), (f x : ℝ) = ((c * x : ℝ) + d : ℝ)} : Set (ℝ → ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q230
find (answer : Set (ℝ → ℝ)) s.t.
: ({f : ℝ → ℝ | Differentiable ℝ f ∧
∀ x : ℝ, ∀ n : ℤ, n > 0 → deriv f x = (f (x + n) - f x)/n} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q230
find_all (answer : Set (ℝ → ℝ)) iff
: ({f : ℝ → ℝ | Differentiable ℝ f ∧
∀ x : ℝ, ∀ n : ℤ, n > 0 → deriv f x = (f (x + n) - f x)/n} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
