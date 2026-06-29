import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $n$ gives a remainder of 3 when divided by 7, then what remainder does $2n+1$ give when divided by 7?

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q158.gt := fun (answer : ℕ) => answer = (0 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q158
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : n % 7 = 3)
: (answer = (2 * n + 1) % 7)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q158
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : n % 7 = 3)
: (answer = (2 * n + 1) % 7)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
