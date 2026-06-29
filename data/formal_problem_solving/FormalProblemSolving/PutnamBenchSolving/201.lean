import FormalProblemSolving.Basic
open Nat Set

/-
## Problem
Find all differentiable functions $f:(0,\infty) \to (0,\infty)$ for which there is a positive real number $a$ such that $f'(\frac{a}{x})=\frac{x}{f(x)}$ for all $x>0$.

## Answer
the set of functions $f$ such that $f(x) = c x^d$ for some $c > 0$ and $d > 0$, with $d = 1$ implying $c = 1$
-/

-- # Formal Answer
noncomputable abbrev Q201.gt := fun (f : ℝ → ℝ) (answer : Set (ℝ → ℝ)) => answer =
  ({f : ℝ → ℝ |
      ∃ c > (0 : ℝ),
        ∃ d > (0 : ℝ),
          (d = (1 : ℝ) → c = (1 : ℝ)) ∧ Set.EqOn f (fun (x : ℝ) => (c * (x ^ d : ℝ) : ℝ)) (Set.Ioi (0 : ℝ) : Set ℝ)} :
    Set (ℝ → ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q201
  (f : ℝ → ℝ)
find (answer : Set (ℝ → ℝ)) s.t.
  (hf : ∀ x > 0, 0 < f x)
  (hf' : DifferentiableOn ℝ f (Ioi 0))
: ((∃ a > 0, ∀ x > 0, deriv f (a / x) = x / f x) ↔ f ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q201
  (f : ℝ → ℝ)
find_all (answer : Set (ℝ → ℝ)) iff
  (hf : ∀ x > 0, 0 < f x)
  (hf' : DifferentiableOn ℝ f (Ioi 0))
: ((∃ a > 0, ∀ x > 0, deriv f (a / x) = x / f x) ↔ f ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
