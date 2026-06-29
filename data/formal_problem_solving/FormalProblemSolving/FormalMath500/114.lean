import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Denali and Nate work for a dog walking business and are paid for each dog they walk. Denali is responsible for $16$ dogs and Nate is responsible for $12$ dogs. Under the company's new policy, they will be assigned or unassigned new dogs in groups of $x$ dogs. The ratio of Denali's pay to Nate's pay would be the same if Denali started walking $4x$ more dogs and Nate stayed at $12$ dogs or if $x$ of Nate's dogs were reassigned to Denali. Find $x$ if $x\neq0$.

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q114.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q114
find (answer : ℕ) s.t.
  (x : ℕ)
  (h₀ : x ≠ 0)
  (Denali_dogs : ℕ)
  (Nate_dogs : ℕ)
  (hx : Denali_dogs = 16 ∧ Nate_dogs = 12)
  (h₁ : (Denali_dogs + 4 * x) / ((1:ℚ) *Nate_dogs) = (Denali_dogs + x) / ((1:ℚ) *(Nate_dogs - x)))
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q114
find_all (answer : ℕ) iff
  (x : ℕ)
  (h₀ : x ≠ 0)
  (Denali_dogs : ℕ)
  (Nate_dogs : ℕ)
  (hx : Denali_dogs = 16 ∧ Nate_dogs = 12)
  (h₁ : (Denali_dogs + 4 * x) / ((1:ℚ) *Nate_dogs) = (Denali_dogs + x) / ((1:ℚ) *(Nate_dogs - x)))
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
