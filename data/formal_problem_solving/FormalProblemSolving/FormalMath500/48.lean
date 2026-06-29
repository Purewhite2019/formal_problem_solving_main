import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The product of three consecutive integers is 120. That product divided by the mean of the three integers is 24. What is the largest of the three consecutive integers?

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q48.gt := fun (answer : ℤ) => answer = (6 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q48
find (answer : ℤ) s.t.
  (a : ℤ)
  (b : ℤ)
  (c : ℤ)
  (h₀ : b = a + 1)
  (h₁ : c = b + 1)
  (h₂ : a * b * c = 120)
  (h₃ : (a * b * c) / ((a + b + c) / 3) = 24)
: (c = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q48
find_all (answer : ℤ) iff
  (a : ℤ)
  (b : ℤ)
  (c : ℤ)
  (h₀ : b = a + 1)
  (h₁ : c = b + 1)
  (h₂ : a * b * c = 120)
  (h₃ : (a * b * c) / ((a + b + c) / 3) = 24)
: (c = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
