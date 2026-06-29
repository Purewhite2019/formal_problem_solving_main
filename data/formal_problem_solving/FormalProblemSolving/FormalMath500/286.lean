import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
When the positive integers with exactly three positive divisors are listed in ascending order, what is the fifth number listed?

## Answer
121
-/

-- # Formal Answer
noncomputable abbrev Q286.gt := fun (answer : ℕ) => answer = (121 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q286
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {n:ℕ | (Nat.divisors n).card = 3})
: (answer ∈ S ∧ (S ∩ {m : ℕ | m ≤ answer}).encard = 5)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q286
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {n:ℕ | (Nat.divisors n).card = 3})
: (answer ∈ S ∧ (S ∩ {m : ℕ | m ≤ answer}).encard = 5)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
