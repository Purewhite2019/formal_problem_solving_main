import FormalProblemSolving.Basic

/-
## Problem
Find all values of $\alpha$ for which the curves $y = \alpha*x^2 + \alpha*x + 1/24$ and $x = \alpha*y^2 + \alpha*y + 1/24$ are tangent to each other.

## Answer
{2 / 3, 3 / 2, (13 + √601) / 12, (13 - √601) / 12}
-/

-- # Formal Answer
noncomputable abbrev Q210.gt := fun (answer : Set ℝ) => answer =
  (Insert.insert (2 / 3 : ℝ)
      (Insert.insert (3 / 2 : ℝ)
          (Insert.insert (((13 : ℝ) + (√(601 : ℝ) : ℝ) : ℝ) / (12 : ℝ) : ℝ)
              ({(((13 : ℝ) - (√(601 : ℝ) : ℝ) : ℝ) / (12 : ℝ) : ℝ)} : Set ℝ) :
            Set ℝ) :
        Set ℝ) :
    Set ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q210
find (answer : Set ℝ) s.t.
  (α : ℝ)
  (P : (ℝ → ℝ) → Prop)
  (P_def : ∀ f, P f ↔ ∃ x y, f x = y ∧ f y = x ∧ deriv f x * deriv f y = 1)
: (P (fun t ↦ α * t ^ 2 + α * t + 1 / 24) ↔ α ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q210
find_all (answer : Set ℝ) iff
  (α : ℝ)
  (P : (ℝ → ℝ) → Prop)
  (P_def : ∀ f, P f ↔ ∃ x y, f x = y ∧ f y = x ∧ deriv f x * deriv f y = 1)
: (P (fun t ↦ α * t ^ 2 + α * t + 1 / 24) ↔ α ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
