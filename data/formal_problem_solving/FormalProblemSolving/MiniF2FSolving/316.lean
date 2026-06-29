import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Jasmine drank 1.5 pints of water on the first 3 miles of her hike. If she continued at this rate, how many pints of water would she drink in the next 10 miles?

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q316.gt := fun (answer : ℝ) => answer = (5 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q316
find (answer : ℝ) s.t.
  (r : ℝ)
  (h_rate : r * 3 = 1.5)
: (answer = r * 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q316
find_all (answer : ℝ) iff
  (r : ℝ)
  (h_rate : r * 3 = 1.5)
: (answer = r * 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
