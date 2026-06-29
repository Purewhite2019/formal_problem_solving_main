import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many zeroes are at the end of $42!$ (42 factorial)?  (Reminder: The number $n!$ is the product of the integers from 1 to $n$.  For example, $5!=5\cdot 4\cdot3\cdot2\cdot 1= 120$.)

## Answer
9
-/

-- # Formal Answer
noncomputable abbrev Q1.gt := fun (answer : ℕ) => answer = (9 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q1
find (answer : ℕ) s.t.
: (∃ n : ℕ, Nat.factorial 42 = n * 10^answer ∧ n % 10 ≠ 0)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q1
find_all (answer : ℕ) iff
: (∃ n : ℕ, Nat.factorial 42 = n * 10^answer ∧ n % 10 ≠ 0)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
