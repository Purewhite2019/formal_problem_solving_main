import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the sum of the digits in the terminating decimal representation of the fraction $\frac{4321}{5^7\cdot2^8}$?

## Answer
14
-/

-- # Formal Answer
noncomputable abbrev Q251.gt := fun (answer : ℕ) => answer = (14 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q251
find (answer : ℕ) s.t.
: (answer = (Nat.digits 10 (4321 * 10^8 / (5^7 * 2^8))).sum)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q251
find_all (answer : ℕ) iff
: (answer = (Nat.digits 10 (4321 * 10^8 / (5^7 * 2^8))).sum)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
