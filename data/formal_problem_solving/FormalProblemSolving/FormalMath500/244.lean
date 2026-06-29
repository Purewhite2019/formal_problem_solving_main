import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The two-digit number $``B6,''$ where $B$ is the tens digit, is the square of a positive integer. How many distinct possible values are there for $B$?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q244.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q244
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {b:ℕ | b > 0 ∧ b ≤ 9 ∧ (∃ x : ℕ, x > 0 ∧ x ≤ 9 ∧ x^2 = 10*b + 6)})
: (answer = S.encard)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q244
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {b:ℕ | b > 0 ∧ b ≤ 9 ∧ (∃ x : ℕ, x > 0 ∧ x ≤ 9 ∧ x^2 = 10*b + 6)})
: (answer = S.encard)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
