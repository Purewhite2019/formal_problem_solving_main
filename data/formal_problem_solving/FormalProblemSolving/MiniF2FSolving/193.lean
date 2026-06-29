import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A rectangular patio has an area of $180$ square feet and a perimeter of $54$ feet. What is the length of the diagonal (in feet) squared?

## Answer
369
-/

-- # Formal Answer
noncomputable abbrev Q193.gt := fun (answer : ℝ) => answer = (369 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q193
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h₁ : a * b = 180)
  (h₂ : 2 * (a + b) = 54)
: (answer = a^2 + b^2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q193
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h₁ : a * b = 180)
  (h₂ : 2 * (a + b) = 54)
: (answer = a^2 + b^2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
