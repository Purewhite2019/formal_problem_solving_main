import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $j$ is a positive integer and the expression $(7j+3)$ is multiplied by 3 and then divided by 7, what is the remainder?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q117.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q117
find (answer : ℕ) s.t.
  (j : ℕ)
  (h₀ : 0 < j)
: (answer = (3 * (7 * j + 3)) % 7)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q117
find_all (answer : ℕ) iff
  (j : ℕ)
  (h₀ : 0 < j)
: (answer = (3 * (7 * j + 3)) % 7)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
