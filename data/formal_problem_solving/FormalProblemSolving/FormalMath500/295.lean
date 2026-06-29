import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A palindrome is a number that reads the same forwards and backwards. The sum of a particular set of three consecutive positive integers is a three-digit palindrome. If the sum is less than 220, what is the greatest possible value for the largest of the three integers in the set?

## Answer
58
-/

-- # Formal Answer
noncomputable abbrev Q295.gt := fun (answer : ℕ) => answer = (58 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q295
find (answer : ℕ) s.t.
: (IsGreatest { L : ℕ | 3 ≤ L ∧ 100 ≤ 3 * (L - 1) ∧ 3 * (L - 1) < 220 ∧ (3 * (L - 1)) / 100 = (3 * (L - 1)) % 10 } answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q295
find_all (answer : ℕ) iff
: (IsGreatest { L : ℕ | 3 ≤ L ∧ 100 ≤ 3 * (L - 1) ∧ 3 * (L - 1) < 220 ∧ (3 * (L - 1)) / 100 = (3 * (L - 1)) % 10 } answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
