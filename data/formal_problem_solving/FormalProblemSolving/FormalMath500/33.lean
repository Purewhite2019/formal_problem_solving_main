import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Yann and Camille go to a restaurant.  If there are 10 items on the menu, and each orders one dish, how many different combinations of meals can Yann and Camille order if they refuse to order the same dish? (It does matter who orders what---Yann ordering chicken and Camille ordering fish is different from Yann ordering fish and Camille ordering chicken.)

## Answer
90
-/

-- # Formal Answer
noncomputable abbrev Q33.gt := fun (answer : ℕ) => answer = (90 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q33
find (answer : ℕ) s.t.
  (total_items : ℕ)
  (combinations : ℕ)
  (h_total_items : 10 = total_items)
  (h_combinations : combinations = total_items * (total_items - 1))
: (combinations = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q33
find_all (answer : ℕ) iff
  (total_items : ℕ)
  (combinations : ℕ)
  (h_total_items : 10 = total_items)
  (h_combinations : combinations = total_items * (total_items - 1))
: (combinations = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
