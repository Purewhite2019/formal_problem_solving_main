import FormalProblemSolving.Basic
open MeasureTheory Set

/-
## Problem
Find every real-valued function $f$ whose domain is an interval $I$ (finite or infinite) having 0 as a left-hand endpoint, such that for every positive member $x$ of $I$ the average of $f$ over the closed interval $[0, x]$ is equal to the geometric mean of the numbers $f(0)$ and $f(x)$.

## Answer
the set of functions $f(x) = \frac{a}{(1 - c x)^2}$ where $a \geq 0$
-/

-- # Formal Answer
noncomputable abbrev Q1.gt := fun (answer : Set (ℝ → ℝ)) => answer =
  ({f : ℝ → ℝ |
      ∃ (a : ℝ) (c : ℝ), a ≥ (0 : ℝ) ∧ f = fun (x : ℝ) => (a / (((1 : ℝ) - (c * x : ℝ) : ℝ) ^ (2 : ℕ) : ℝ) : ℝ)} :
    Set (ℝ → ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q1
find (answer : Set (ℝ → ℝ)) s.t.
  (P : Set ℝ → (ℝ → ℝ) → Prop)
  (P_def : ∀ s f, P s f ↔ 0 ≤ f ∧ ∀ x ∈ s, ⨍ t in Ico 0 x, f t = √(f 0 * f x))
: ((∀ f,
      (P (Ioi 0) f → ∃ g ∈ answer, EqOn f g (Ici 0)) ∧
      (∀ e > 0, P (Ioo 0 e) f → ∃ g ∈ answer, EqOn f g (Ico 0 e))) ∧
    ∀ f ∈ answer, P (Ioi 0) f ∨ (∃ e > 0, P (Ioo 0 e) f))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q1
find_all (answer : Set (ℝ → ℝ)) iff
  (P : Set ℝ → (ℝ → ℝ) → Prop)
  (P_def : ∀ s f, P s f ↔ 0 ≤ f ∧ ∀ x ∈ s, ⨍ t in Ico 0 x, f t = √(f 0 * f x))
: ((∀ f,
      (P (Ioi 0) f → ∃ g ∈ answer, EqOn f g (Ici 0)) ∧
      (∀ e > 0, P (Ioo 0 e) f → ∃ g ∈ answer, EqOn f g (Ico 0 e))) ∧
    ∀ f ∈ answer, P (Ioi 0) f ∨ (∃ e > 0, P (Ioo 0 e) f))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
