import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
One morning each member of Angela's family drank an 8-ounce mixture of coffee with milk. The amounts of coffee and milk varied from cup to cup, but were never zero. Angela drank a quarter of the total amount of milk and a sixth of the total amount of coffee. How many people are in the family?

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q226.gt := fun (answer : ℕ+) => answer = (5 : ℕ+)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q226
find (answer : ℕ+) s.t.
  (x : ℕ+)
  (y : ℕ+)
  (n : ℕ+)
  (h_total : (x : ℝ) + (y : ℝ) = 8 * (n : ℝ))
  (h_angela : (x : ℝ) / 4 + (y : ℝ) / 6 = 8)
: (answer = n)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q226
find_all (answer : ℕ+) iff
  (x : ℕ+)
  (y : ℕ+)
  (n : ℕ+)
  (h_total : (x : ℝ) + (y : ℝ) = 8 * (n : ℝ))
  (h_angela : (x : ℝ) / 4 + (y : ℝ) / 6 = 8)
: (answer = n)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
