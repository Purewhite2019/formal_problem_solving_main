import FormalProblemSolving.Basic

/-
## Problem
For which real numbers $k$ does there exist a continuous function $f : \mathbb{R} \to \mathbb{R}$ such that $f(f(x)) = kx^9$ for all real $x$?

## Answer
$k \geq 0$
-/

-- # Formal Answer
noncomputable abbrev Q56.gt := fun (k : ℝ) (answer : ℝ → Prop) => answer = fun (k : ℝ) => k ≥ (0 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q56
  (k : ℝ)
find (answer : ℝ → Prop) s.t.
: (answer k ↔ ∃ f : ℝ → ℝ, Continuous f ∧ ∀ x : ℝ, f (f x) = k * x^9)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q56
  (k : ℝ)
find_all (answer : ℝ → Prop) iff
: (answer k ↔ ∃ f : ℝ → ℝ, Continuous f ∧ ∀ x : ℝ, f (f x) = k * x^9)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
