import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the modulo $6$ remainder of the sum $1+2+3+4+\dots+98+99+100?$

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q198.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q198
find (answer : ℕ) s.t.
: (answer = (∑ k in (Finset.range 101), k) % 6)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q198
find_all (answer : ℕ) iff
: (answer = (∑ k in (Finset.range 101), k) % 6)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
