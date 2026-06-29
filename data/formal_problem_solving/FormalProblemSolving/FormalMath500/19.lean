import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many positive two-digit integers are factors of both 100 and 150?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q19.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q19
find (answer : ℕ) s.t.
: (answer = ((Nat.divisors (Nat.gcd 100 150)).filter fun x => 10 ≤ x ∧ x ≤ 99).card)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q19
find_all (answer : ℕ) iff
: (answer = ((Nat.divisors (Nat.gcd 100 150)).filter fun x => 10 ≤ x ∧ x ≤ 99).card)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
