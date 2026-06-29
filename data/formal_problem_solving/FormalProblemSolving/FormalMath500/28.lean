import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A robot moving forward at a constant speed takes 2.5 hours to travel 1 kilometer. Moving forward at this same constant speed, it takes the robot 90 seconds to travel the length of a particular hallway. How many meters long is the hallway?

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q28.gt := fun (answer : ℝ) => answer = (10 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q28
find (answer : ℝ) s.t.
  (speed : ℝ)
  (distance_hallway : ℝ)
  (h₀ : speed = 1 / 2.5)
  (h₁ : distance_hallway = speed * (90 / 3600))
: (distance_hallway * 1000 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q28
find_all (answer : ℝ) iff
  (speed : ℝ)
  (distance_hallway : ℝ)
  (h₀ : speed = 1 / 2.5)
  (h₁ : distance_hallway = speed * (90 / 3600))
: (distance_hallway * 1000 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
