import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The volume of a cone is given by the formula $V = \frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume?

## Answer
65
-/

-- # Formal Answer
noncomputable abbrev Q172.gt := fun (answer : ℝ) => answer = (65 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q172
find (answer : ℝ) s.t.
  (B : ℝ)
  (h : ℝ)
  (hB : B = 30)
  (hh : h = 6.5)
: (answer = (1/3 : ℝ) * B * h)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q172
find_all (answer : ℝ) iff
  (B : ℝ)
  (h : ℝ)
  (hB : B = 30)
  (hh : h = 6.5)
: (answer = (1/3 : ℝ) * B * h)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
