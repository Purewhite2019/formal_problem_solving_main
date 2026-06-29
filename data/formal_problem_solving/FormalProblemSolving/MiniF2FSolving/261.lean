import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the least positive integer $n$ such that $80325$ divides $n!$?

## Answer
17
-/

-- # Formal Answer
noncomputable abbrev Q261.gt := fun (answer : ℕ) => answer = (17 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q261
find (answer : ℕ) s.t.
: (0 < answer) ∧ (80325 ∣ answer !) ∧ (∀ n : ℕ, 0 < n → 80325 ∣ n ! → answer ≤ n)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q261
find_all (answer : ℕ) iff
: (0 < answer) ∧ (80325 ∣ answer !) ∧ (∀ n : ℕ, 0 < n → 80325 ∣ n ! → answer ≤ n)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
