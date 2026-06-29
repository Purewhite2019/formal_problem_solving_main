import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A polynomial with integer coefficients is of the form
\[2x^4 + a_3 x^3 + a_2 x^2 + a_1 x + 1 = 0.\]Find the number of different possible rational roots of this polynomial.

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q336.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q336
find (answer : ℕ) s.t.
: (answer = Set.ncard {x : ℚ | ∃ (a1 a2 a3 : ℤ), 2 * x ^ 4 + a3 * x ^ 3 + a2 * x ^ 2 + a1 * x + 1 = 0})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q336
find_all (answer : ℕ) iff
: (answer = Set.ncard {x : ℚ | ∃ (a1 a2 a3 : ℤ), 2 * x ^ 4 + a3 * x ^ 3 + a2 * x ^ 2 + a1 * x + 1 = 0})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
