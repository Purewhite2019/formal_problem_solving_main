import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Jim and Martha are standing together at the corner of a rectangular field. Jim walks diagonally across the field. Martha gets to the same location by walking along its length and width. The field is 300 feet wide and 400 feet long. How many feet less than Martha does Jim walk?

## Answer
200
-/

-- # Formal Answer
noncomputable abbrev Q49.gt := fun (answer : ℝ) => answer = (200 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q49
find (answer : ℝ) s.t.
  (width : ℝ)
  (length : ℝ)
  (jim_distance : ℝ)
  (martha_distance : ℝ)
  (h_width : width = 300)
  (h_length : length = 400)
  (h_jim : jim_distance = Real.sqrt (width^2 + length^2))
  (h_martha : martha_distance = width + length)
: (answer = martha_distance - jim_distance)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q49
find_all (answer : ℝ) iff
  (width : ℝ)
  (length : ℝ)
  (jim_distance : ℝ)
  (martha_distance : ℝ)
  (h_width : width = 300)
  (h_length : length = 400)
  (h_jim : jim_distance = Real.sqrt (width^2 + length^2))
  (h_martha : martha_distance = width + length)
: (answer = martha_distance - jim_distance)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
