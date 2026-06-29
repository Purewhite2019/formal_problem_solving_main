import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Express $5^5\div5^4-5^3+5^2\cdot5^1$ as an integer.

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q64.gt := fun (answer : ℤ) => answer = (5 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q64
find (answer : ℤ) s.t.
: ((5^5 / 5^4 - 5^3 + 5^2 * 5^1) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q64
find_all (answer : ℤ) iff
: ((5^5 / 5^4 - 5^3 + 5^2 * 5^1) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
