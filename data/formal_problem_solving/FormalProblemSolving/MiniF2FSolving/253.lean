import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sum of two numbers is 25 and their difference is 11. What is the larger of the two numbers?

## Answer
18
-/

-- # Formal Answer
noncomputable abbrev Q253.gt := fun (answer : ℝ) => answer = (18 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q253
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : x + y = 25)
  (h₁ : x - y = 11)
: (answer = max x y)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q253
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : x + y = 25)
  (h₁ : x - y = 11)
: (answer = max x y)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
