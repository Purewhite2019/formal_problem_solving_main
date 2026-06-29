import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Tim is doing a cakewalk with $11$ steps. He takes his first step on step $1$ and takes a total of $139$ steps, walking in a circle (so that after the 11th step he reaches the first step again). Which step of the cakewalk does he end on?

## Answer
7
-/

-- # Formal Answer
noncomputable abbrev Q39.gt := fun (answer : ℕ) => answer = (7 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q39
find (answer : ℕ) s.t.
  (num_steps : ℕ)
  (total : ℕ)
  (h_num_steps : num_steps = 11)
  (h_total : total = 139)
: (answer = ((total - 1) % num_steps) + 1)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q39
find_all (answer : ℕ) iff
  (num_steps : ℕ)
  (total : ℕ)
  (h_num_steps : num_steps = 11)
  (h_total : total = 139)
: (answer = ((total - 1) % num_steps) + 1)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
