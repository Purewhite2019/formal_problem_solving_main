import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
In a rectangular coordinate system, the line $3y = x$ intersects the line $2x + 5y = 11$ at point $A$. What is the sum of the coordinates of point $A$?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q287.gt := fun (answer : ℝ) => answer = (4 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q287
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : 3 * y = x)
  (h₁ : 2 * x + 5 * y = 11)
: (answer = x + y)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q287
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : 3 * y = x)
  (h₁ : 2 * x + 5 * y = 11)
: (answer = x + y)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
