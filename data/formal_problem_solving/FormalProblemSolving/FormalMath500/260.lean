import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $\frac{a}{b}$ is the probability that the reciprocal of a randomly selected positive odd integer less than 2010 gives a terminating decimal, with $a$ and $b$ being relatively prime positive integers, what is $a+b$?

## Answer
202
-/

-- # Formal Answer
noncomputable abbrev Q260.gt := fun (answer : ℕ) => answer = (202 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q260
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (hab : a * 1005 = b * (Int.floor ((Nat.log 5 2010):ℝ) + 1))
  (hab2 : Nat.gcd a b = 1)
: (answer = a + b)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q260
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (hab : a * 1005 = b * (Int.floor ((Nat.log 5 2010):ℝ) + 1))
  (hab2 : Nat.gcd a b = 1)
: (answer = a + b)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
