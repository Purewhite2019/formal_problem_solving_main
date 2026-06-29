import FormalProblemSolving.Basic
open Function

/-
## Problem
Let $c>0$ be a constant. Give a complete description, with proof, of the set of all continuous functions $f:\mathbb{R} \to \mathbb{R}$ such that $f(x)=f(x^2+c)$ for all $x \in \mathbb{R}$.

## Answer
the set of functions described by the condition
-/

-- # Formal Answer
noncomputable abbrev Q164.gt := fun (c : ℝ) (f : ℝ → ℝ) (answer : ℝ → Set (ℝ → ℝ)) => answer = fun (c : ℝ) =>
  (if c ≤ (1 / 4 : ℝ) then ({f : ℝ → ℝ | ∃ (d : ℝ), ∀ (x : ℝ), (f x : ℝ) = d} : Set (ℝ → ℝ))
    else
      ({f : ℝ → ℝ |
          ContinuousOn f (Set.Icc (0 : ℝ) c : Set ℝ) ∧
            (f (0 : ℝ) : ℝ) = (f c : ℝ) ∧
              (∀ (x : ℝ), x > (0 : ℝ) → (f x : ℝ) = (f ((x ^ (2 : ℕ) : ℝ) + c : ℝ) : ℝ)) ∧
                ∀ (x : ℝ), x < (0 : ℝ) → (f x : ℝ) = (f (-x : ℝ) : ℝ)} :
        Set (ℝ → ℝ)) :
    Set (ℝ → ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q164
  (c : ℝ)
  (f : ℝ → ℝ)
find (answer : ℝ → Set (ℝ → ℝ)) s.t.
  (cgt0 : c > 0)
: ((Continuous f ∧ ∀ x : ℝ, f x = f (x ^ 2 + c)) ↔ f ∈ answer c)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q164
  (c : ℝ)
  (f : ℝ → ℝ)
find_all (answer : ℝ → Set (ℝ → ℝ)) iff
  (cgt0 : c > 0)
: ((Continuous f ∧ ∀ x : ℝ, f x = f (x ^ 2 + c)) ↔ f ∈ answer c)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
