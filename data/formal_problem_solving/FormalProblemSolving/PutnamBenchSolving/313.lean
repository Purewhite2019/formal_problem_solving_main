import FormalProblemSolving.Basic
open Polynomial

/-
## Problem
Find all continuous functions $f:\mathbb{R}^+ \to \mathbb{R}^+$ such that $f(xf(y))+f(yf(x))=1+f(x+y)$ for all $x,y>0$.

## Answer
the set of functions of the form f(x) = 1 / (1 + c * x) where c ≥ 0
-/

-- # Formal Answer
noncomputable abbrev Q313.gt := fun (f : Set.Ioi (0 : ℝ) → Set.Ioi (0 : ℝ)) (answer : Set (Set.Ioi (0 : ℝ) → Set.Ioi (0 : ℝ))) => answer =
  ({f : ↑(Set.Ioi (0 : ℝ) : Set ℝ) → ↑(Set.Ioi (0 : ℝ) : Set ℝ) |
      ∃ c ≥ (0 : ℝ),
        ∀ (x : ↑(Set.Ioi (0 : ℝ) : Set ℝ)),
          (↑(f x : ↑(Set.Ioi (0 : ℝ) : Set ℝ)) : ℝ) = ((1 : ℝ) / ((1 : ℝ) + (c * (↑x : ℝ) : ℝ) : ℝ) : ℝ)} :
    Set (↑(Set.Ioi (0 : ℝ) : Set ℝ) → ↑(Set.Ioi (0 : ℝ) : Set ℝ)))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q313
  (f : Set.Ioi (0 : ℝ) → Set.Ioi (0 : ℝ))
find (answer : Set (Set.Ioi (0 : ℝ) → Set.Ioi (0 : ℝ))) s.t.
  (hf : Continuous f)
: ((∃ fr : ℝ → ℝ, (∀ x : Set.Ioi (0 : ℝ), fr x = f x) ∧ (∀ x y : Set.Ioi (0 : ℝ), fr (x * fr y) + fr (y * fr x) = 1 + fr (x + y))) ↔ f ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q313
  (f : Set.Ioi (0 : ℝ) → Set.Ioi (0 : ℝ))
find_all (answer : Set (Set.Ioi (0 : ℝ) → Set.Ioi (0 : ℝ))) iff
  (hf : Continuous f)
: ((∃ fr : ℝ → ℝ, (∀ x : Set.Ioi (0 : ℝ), fr x = f x) ∧ (∀ x y : Set.Ioi (0 : ℝ), fr (x * fr y) + fr (y * fr x) = 1 + fr (x + y))) ↔ f ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
