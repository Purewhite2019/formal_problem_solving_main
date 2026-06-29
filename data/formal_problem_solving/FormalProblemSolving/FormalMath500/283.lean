import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $A$ is the sum of the positive divisors of $500$, what is the sum of the distinct prime divisors of $A$?

## Answer
25
-/

-- # Formal Answer
noncomputable abbrev Q283.gt := fun (answer : ℕ) => answer = (25 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q283
find (answer : ℕ) s.t.
  (A : ℕ)
  (hA : A = ∑ n in Nat.divisors 500, n)
: (answer = ∑ p in Nat.primeFactors A, p)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q283
find_all (answer : ℕ) iff
  (A : ℕ)
  (hA : A = ∑ n in Nat.divisors 500, n)
: (answer = ∑ p in Nat.primeFactors A, p)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
