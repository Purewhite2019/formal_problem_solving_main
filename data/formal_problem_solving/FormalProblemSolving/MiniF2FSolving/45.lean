import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For each positive integer $n$, let $n!$ denote the product $1\cdot 2\cdot 3\cdot\,\cdots\,\cdot (n-1)\cdot n$.

What is the remainder when $9!$ is divided by $10$?

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q45.gt := fun (answer : ℕ) => answer = (0 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q45
find (answer : ℕ) s.t.
: (answer = 9! % 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q45
find_all (answer : ℕ) iff
: (answer = 9! % 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
