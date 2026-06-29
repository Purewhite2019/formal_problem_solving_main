import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the remainder when $91145 + 91146 + 91147 + 91148$ is divided by 4.

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q79.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q79
find (answer : ℕ) s.t.
  (h_sum : ℕ)
  (h_sum_def : h_sum = 91145 + 91146 + 91147 + 91148)
: (answer = h_sum % 4)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q79
find_all (answer : ℕ) iff
  (h_sum : ℕ)
  (h_sum_def : h_sum = 91145 + 91146 + 91147 + 91148)
: (answer = h_sum % 4)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
