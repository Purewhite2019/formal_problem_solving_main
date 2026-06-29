import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the largest negative integer $x$ satisfying $$24x \equiv 15 \pmod{1199}~?$$

## Answer
-449
-/

-- # Formal Answer
noncomputable abbrev Q228.gt := fun (answer : ℤ) => answer = (-449 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q228
find (answer : ℤ) s.t.
  (S : Set ℤ)
  (hS : S = {x : ℤ | x < 0 ∧ (24 * x) % 1199 = 15})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q228
find_all (answer : ℤ) iff
  (S : Set ℤ)
  (hS : S = {x : ℤ | x < 0 ∧ (24 * x) % 1199 = 15})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
