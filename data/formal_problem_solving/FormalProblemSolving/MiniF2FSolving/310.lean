import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Today a father's age is five times his son's age. Exactly three years ago, the sum of their ages was 30. How old is the son today?

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q310.gt := fun (answer : ℕ) => answer = (6 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q310
find (answer : ℕ) s.t.
  (son_age : ℕ)
  (father_age : ℕ)
  (h₀ : 0 < son_age ∧ 0 < father_age)
  (h₁ : father_age = 5 * son_age)
  (h₂ : (son_age - 3) + (father_age - 3) = 30)
: (answer = son_age)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q310
find_all (answer : ℕ) iff
  (son_age : ℕ)
  (father_age : ℕ)
  (h₀ : 0 < son_age ∧ 0 < father_age)
  (h₁ : father_age = 5 * son_age)
  (h₂ : (son_age - 3) + (father_age - 3) = 30)
: (answer = son_age)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
