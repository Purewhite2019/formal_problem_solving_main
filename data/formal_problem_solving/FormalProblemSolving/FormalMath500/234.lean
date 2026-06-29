import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The proper divisors of 12 are 1, 2, 3, 4 and 6. A proper divisor of an integer $N$ is a positive divisor of $N$ that is less than $N$. What is the sum of the proper divisors of the sum of the proper divisors of 284?

## Answer
284
-/

-- # Formal Answer
noncomputable abbrev Q234.gt := fun (answer : ℕ) => answer = (284 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q234
find (answer : ℕ) s.t.
  (sum : ℕ)
  (h_sum : sum = ∑ x in Nat.properDivisors 284, x)
: (answer = ∑ d in Nat.properDivisors sum, d)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q234
find_all (answer : ℕ) iff
  (sum : ℕ)
  (h_sum : sum = ∑ x in Nat.properDivisors 284, x)
: (answer = ∑ d in Nat.properDivisors sum, d)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
