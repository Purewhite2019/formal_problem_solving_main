import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the smallest positive integer $N$ such that the value $7 + (30 \times N)$ is not a prime number?

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q366.gt := fun (answer : ℕ) => answer = (6 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q366
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {n | ¬ Nat.Prime (7 + 30 * n)})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q366
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {n | ¬ Nat.Prime (7 + 30 * n)})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
