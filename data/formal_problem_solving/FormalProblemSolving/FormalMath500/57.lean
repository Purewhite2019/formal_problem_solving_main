import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A bicycle is traveling at 20 feet per minute. What is the bicycle's speed expressed in inches per second?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q57.gt := fun (answer : ℝ) => answer = (4 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q57
find (answer : ℝ) s.t.
  (speed_feet_per_minute : ℝ)
  (h_speed_feet_per_minute : speed_feet_per_minute = 20)
  (inches_per_foot : ℝ)
  (h_inches_per_foot : inches_per_foot = 12)
  (seconds_per_minute : ℝ)
  (h_seconds_per_minute : seconds_per_minute = 60)
  (speed_inches_per_second : ℝ)
  (h_speed_inches_per_second : speed_inches_per_second = speed_feet_per_minute * inches_per_foot / seconds_per_minute)
: (speed_inches_per_second = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q57
find_all (answer : ℝ) iff
  (speed_feet_per_minute : ℝ)
  (h_speed_feet_per_minute : speed_feet_per_minute = 20)
  (inches_per_foot : ℝ)
  (h_inches_per_foot : inches_per_foot = 12)
  (seconds_per_minute : ℝ)
  (h_seconds_per_minute : seconds_per_minute = 60)
  (speed_inches_per_second : ℝ)
  (h_speed_inches_per_second : speed_inches_per_second = speed_feet_per_minute * inches_per_foot / seconds_per_minute)
: (speed_inches_per_second = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
