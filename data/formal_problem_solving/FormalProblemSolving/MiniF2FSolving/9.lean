import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the greatest common factor of $20 !$ and $200,\!000$?  (Reminder: If $n$ is a positive integer, then $n!$ stands for the product $1\cdot 2\cdot 3\cdot \cdots \cdot (n-1)\cdot n$.)

## Answer
40,000
-/

-- # Formal Answer
noncomputable abbrev Q9.gt := fun (answer : ℕ) => answer = (40000 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q9
find (answer : ℕ) s.t.
: (answer = Nat.gcd (Nat.factorial 20) 200000)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q9
find_all (answer : ℕ) iff
: (answer = Nat.gcd (Nat.factorial 20) 200000)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
