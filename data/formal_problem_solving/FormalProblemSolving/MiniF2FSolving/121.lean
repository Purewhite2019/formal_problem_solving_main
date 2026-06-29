import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the remainder when $7!$ is divided by $23$?

(Note: $n!$ means "n factorial", defined as the product of all integers from $1$ to $n$.)

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q121.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q121
find (answer : ℕ) s.t.
: (answer = 7! % 23)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q121
find_all (answer : ℕ) iff
: (answer = 7! % 23)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
