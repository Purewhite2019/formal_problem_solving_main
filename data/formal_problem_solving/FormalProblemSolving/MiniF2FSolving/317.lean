import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Sally, Wei-Hwa, and Zoe are playing a game of marbles involving first arranging as many piles of 10 marbles as possible.  Sally brought 239 marbles, Wei-Hwa brought 174 marbles, and Zoe brought 83 marbles.  If all their marbles are grouped together, how many must be removed in order to start the game?

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q317.gt := fun (answer : ℕ) => answer = (6 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q317
find (answer : ℕ) s.t.
: (answer = (239 + 174 + 83) % 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q317
find_all (answer : ℕ) iff
: (answer = (239 + 174 + 83) % 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
