import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the least positive integer multiple of 30 that can be written with only the digits 0 and 2?

## Answer
2220
-/

-- # Formal Answer
noncomputable abbrev Q269.gt := fun (answer : ℕ) => answer = (2220 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q269
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {x : ℕ | 0 < x ∧ 30 ∣ x ∧ ∀ d ∈ Nat.digits 10 x, d = 0 ∨ d = 2})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q269
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {x : ℕ | 0 < x ∧ 30 ∣ x ∧ ∀ d ∈ Nat.digits 10 x, d = 0 ∨ d = 2})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
