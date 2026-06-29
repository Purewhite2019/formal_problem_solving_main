import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Notice that  \[35\cdot40=1400.\]Find some integer $n$ with $0\leq n<1399$ such that $n$ is the multiplicative inverse to 160 modulo 1399.

## Answer
1058
-/

-- # Formal Answer
noncomputable abbrev Q284.gt := fun (answer : ZMod 1399) => answer = (1058 : ZMod (1399 : ℕ))

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q284
find (answer : ZMod 1399) s.t.
: (answer = 160⁻¹)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q284
find_all (answer : ZMod 1399) iff
: (answer = 160⁻¹)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
