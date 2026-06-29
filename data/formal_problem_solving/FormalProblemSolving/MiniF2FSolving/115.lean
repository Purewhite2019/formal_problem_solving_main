import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
When all the girls at Madeline's school line up in rows of eight, there are seven left over.

If instead they line up in rows of four, how many are left over?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q115.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q115
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : n % 8 = 7)
: (answer = n % 4)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q115
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : n % 8 = 7)
: (answer = n % 4)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
