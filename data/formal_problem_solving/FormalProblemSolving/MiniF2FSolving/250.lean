import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A $3491$ by $3491$ square has its length decreased by $60$ and its width increased by $60$.  By how much does its area change?

## Answer
3600
-/

-- # Formal Answer
noncomputable abbrev Q250.gt := fun (answer : ℤ) => answer = (3600 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q250
find (answer : ℤ) s.t.
  (original_area : ℤ)
  (new_area : ℤ)
  (h_area : new_area = (3491 - 60)*(3491 + 60) ∧ original_area = 3491 * 3491)
  (area_change : ℤ)
  (h_c : area_change = new_area - original_area)
: (answer = abs area_change)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q250
find_all (answer : ℤ) iff
  (original_area : ℤ)
  (new_area : ℤ)
  (h_area : new_area = (3491 - 60)*(3491 + 60) ∧ original_area = 3491 * 3491)
  (area_change : ℤ)
  (h_c : area_change = new_area - original_area)
: (answer = abs area_change)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
