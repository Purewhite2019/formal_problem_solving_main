import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $0.\overline{1331}$ is written as a fraction $\frac{a}{b}$ with $a$ and $b$ being positive integers with a greatest common divisor of 1, what is $a+b$?

## Answer
1030
-/

-- # Formal Answer
noncomputable abbrev Q274.gt := fun (answer : ℕ) => answer = (1030 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q274
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (hab : 1331 * b  =  9999 * a)
  (hab2 : Nat.gcd a b = 1)
: (answer = a + b)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q274
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (hab : 1331 * b  =  9999 * a)
  (hab2 : Nat.gcd a b = 1)
: (answer = a + b)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
