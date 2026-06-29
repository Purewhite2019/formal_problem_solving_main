import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For how many two-digit primes is the sum of the digits equal to 8?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q53.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q53
find (answer : ℕ) s.t.
  (S : Set (ℕ × ℕ))
  (hS : S = {x : ℕ × ℕ | Nat.Prime (x.1*10 + x.2) ∧ x.1 > 0 ∧ x.1 ≤ 9 ∧ x.2 ≥ 0 ∧ x.2 ≤ 9 ∧ x.1 + x.2 = 8})
: (answer = S.encard)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q53
find_all (answer : ℕ) iff
  (S : Set (ℕ × ℕ))
  (hS : S = {x : ℕ × ℕ | Nat.Prime (x.1*10 + x.2) ∧ x.1 > 0 ∧ x.1 ≤ 9 ∧ x.2 ≥ 0 ∧ x.2 ≤ 9 ∧ x.1 + x.2 = 8})
: (answer = S.encard)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
