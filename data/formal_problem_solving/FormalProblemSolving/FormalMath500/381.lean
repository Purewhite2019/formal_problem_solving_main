import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find all the integer roots of
\[x^4 + 5x^3 + 9x^2 - x - 14 = 0.\]Enter all the integer roots, separated by commas.

## Answer
-2,1
-/

-- # Formal Answer
noncomputable abbrev Q381.gt := fun (answer : Set ℤ) => answer = (Insert.insert (-2 : ℤ) ({(1 : ℤ)} : Set ℤ) : Set ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q381
find (answer : Set ℤ) s.t.
: (answer = {x : ℤ | x^4 + 5 * x^3 + 9 * x^2 - x - 14 = 0})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q381
find_all (answer : Set ℤ) iff
: (answer = {x : ℤ | x^4 + 5 * x^3 + 9 * x^2 - x - 14 = 0})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
