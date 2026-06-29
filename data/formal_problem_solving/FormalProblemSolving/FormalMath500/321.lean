import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the roots of $(x - 3)^3 + (x -7)^3 = (2x - 10)^3.$

## Answer
3, 5, 7
-/

-- # Formal Answer
noncomputable abbrev Q321.gt := fun (answer : Set ℝ) => answer = (Insert.insert (3 : ℝ) (Insert.insert (5 : ℝ) ({(7 : ℝ)} : Set ℝ) : Set ℝ) : Set ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q321
find (answer : Set ℝ) s.t.
: (answer = {x : ℝ | (x - 3)^3 + (x - 7)^3 = (2 * x - 10)^3})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q321
find_all (answer : Set ℝ) iff
: (answer = {x : ℝ | (x - 3)^3 + (x - 7)^3 = (2 * x - 10)^3})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
