import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
My father's age is $1222_{3}$, in base three to represent his three lower limbs -- two legs and a cane.  How old is he in base ten?

## Answer
53
-/

-- # Formal Answer
noncomputable abbrev Q327.gt := fun (answer : ℕ) => answer = (53 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q327
find (answer : ℕ) s.t.
: (answer = 1 * 3^3 + 2 * 3^2 + 2 * 3 + 2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q327
find_all (answer : ℕ) iff
: (answer = 1 * 3^3 + 2 * 3^2 + 2 * 3 + 2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
