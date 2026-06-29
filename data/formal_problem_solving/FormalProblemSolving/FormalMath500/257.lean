import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many of the same digits are found in the base 7 and base 8 representations of $629_{10}$? For example, $121_{3}$ and $413_{5}$ would have one digit in common.

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q257.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q257
find (answer : ℕ) s.t.
: (answer = ((Nat.digits 7 629).toFinset ∩ (Nat.digits 8 629).toFinset).card)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q257
find_all (answer : ℕ) iff
: (answer = ((Nat.digits 7 629).toFinset ∩ (Nat.digits 8 629).toFinset).card)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
