import FormalProblemSolving.Basic
open Set

/-
## Problem
Let $\alpha$ be a real number. Find all continuous real-valued functions $f : [0, 1] \to (0, \infty)$ such that
\begin{align*}
\int_0^1 f(x) dx &= 1, \\
\int_0^1 x f(x) dx &= \alpha, \\
\int_0^1 x^2 f(x) dx &= \alpha^2. \\
\end{align*}

## Answer
the empty set
-/

-- # Formal Answer
noncomputable abbrev Q7.gt := fun (answer : ℝ → Set (ℝ → ℝ)) => answer = fun (x : ℝ) => (∅ : Set (ℝ → ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q7
find (answer : ℝ → Set (ℝ → ℝ)) s.t.
  (α : ℝ)
: (answer α = {f : ℝ → ℝ | (∀ x ∈ Icc 0 1, f x > 0) ∧ ContinuousOn f (Icc 0 1) ∧ ∫ x in (0)..1, f x = 1 ∧ ∫ x in (0)..1, x * f x = α ∧ ∫ x in (0)..1, x^2 * f x = α^2})
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q7
find_all (answer : ℝ → Set (ℝ → ℝ)) iff
  (α : ℝ)
: (answer α = {f : ℝ → ℝ | (∀ x ∈ Icc 0 1, f x > 0) ∧ ContinuousOn f (Icc 0 1) ∧ ∫ x in (0)..1, f x = 1 ∧ ∫ x in (0)..1, x * f x = α ∧ ∫ x in (0)..1, x^2 * f x = α^2})
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
