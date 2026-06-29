import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose $t$ is a positive integer such that $\mathop{\text{lcm}}[12,t]^3=(12t)^2$. What is the smallest possible value for $t$?

## Answer
18
-/

-- # Formal Answer
noncomputable abbrev Q185.gt := fun (answer : ℕ) => answer = (18 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q185
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {t | 0 < t ∧ (Nat.lcm 12 t)^3 = (12 * t)^2})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q185
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {t | 0 < t ∧ (Nat.lcm 12 t)^3 = (12 * t)^2})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
