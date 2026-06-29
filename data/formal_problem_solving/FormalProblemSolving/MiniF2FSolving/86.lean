import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Determine the largest possible integer $n$ such that $942!$ is divisible by $15^n$.

## Answer
233
-/

-- # Formal Answer
noncomputable abbrev Q86.gt := fun (answer : ℕ) => answer = (233 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q86
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {n : ℕ | 15^n ∣ 942!})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q86
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {n : ℕ | 15^n ∣ 942!})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
