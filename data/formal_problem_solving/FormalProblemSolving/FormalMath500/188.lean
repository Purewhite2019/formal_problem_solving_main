import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $2^8=4^x$, what is the value of $x$?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q188.gt := fun (answer : ℝ) => answer = (4 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q188
find (answer : ℝ) s.t.
  (x : ℝ)
  (h_eq : (2 : ℝ) ^ 8 = (4 : ℝ) ^ x)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q188
find_all (answer : ℝ) iff
  (x : ℝ)
  (h_eq : (2 : ℝ) ^ 8 = (4 : ℝ) ^ x)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
