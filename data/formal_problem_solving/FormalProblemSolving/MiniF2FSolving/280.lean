import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given the equations $3x+y=17,5y+z=14$ and $3x+5z=41$, what is the value of the sum $x+y+z$?

## Answer
12
-/

-- # Formal Answer
noncomputable abbrev Q280.gt := fun (answer : ℝ) => answer = (12 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q280
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h₀ : 3 * x + y = 17)
  (h₁ : 5 * y + z = 14)
  (h₂ : 3 * x + 5 * z = 41)
: (answer = x + y + z)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q280
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h₀ : 3 * x + y = 17)
  (h₁ : 5 * y + z = 14)
  (h₂ : 3 * x + 5 * z = 41)
: (answer = x + y + z)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
