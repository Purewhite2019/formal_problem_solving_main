import FormalProblemSolving.Basic
open Topology Filter Polynomial

/-
## Problem
Find every twice-differentiable real-valued function $f$ with domain the set of all real numbers and satisfying the functional equation $(f(x))^2-(f(y))^2=f(x+y)f(x-y)$ for all real numbers $x$ and $y$.

## Answer
the set of functions of the form $A \sinh(k u)$, $A u$, or $A \sin(k u)$
-/

-- # Formal Answer
noncomputable abbrev Q6.gt := fun (answer : Set (ℝ → ℝ)) => answer =
  ((({x : ℝ → ℝ | ∃ (A : ℝ) (k : ℝ), (fun (u : ℝ) => (A * (Real.sinh (k * u : ℝ) : ℝ) : ℝ)) = x} : Set (ℝ → ℝ)) ∪
          ({x : ℝ → ℝ | ∃ (A : ℝ), (fun (u : ℝ) => (A * u : ℝ)) = x} : Set (ℝ → ℝ)) :
        Set (ℝ → ℝ)) ∪
      ({x : ℝ → ℝ | ∃ (A : ℝ) (k : ℝ), (fun (u : ℝ) => (A * (Real.sin (k * u : ℝ) : ℝ) : ℝ)) = x} : Set (ℝ → ℝ)) :
    Set (ℝ → ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q6
find (answer : Set (ℝ → ℝ)) s.t.
  (f : ℝ → ℝ)
: (f ∈ answer ↔
      (ContDiff ℝ 1 f ∧ Differentiable ℝ (deriv f) ∧
      ∀ x y : ℝ, (f x) ^ 2 - (f y) ^ 2 = f (x + y) * f (x - y)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q6
find_all (answer : Set (ℝ → ℝ)) iff
  (f : ℝ → ℝ)
: (f ∈ answer ↔
      (ContDiff ℝ 1 f ∧ Differentiable ℝ (deriv f) ∧
      ∀ x y : ℝ, (f x) ^ 2 - (f y) ^ 2 = f (x + y) * f (x - y)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
