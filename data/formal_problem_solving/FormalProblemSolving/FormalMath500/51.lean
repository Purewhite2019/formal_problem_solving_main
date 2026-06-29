import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Charlie is riding on his unicycle. If the unicycle's wheel has a radius of 9 inches and makes 2 revolutions every 3 seconds, what is the unicycle's speed in inches per second?

## Answer
12\pi
-/

-- # Formal Answer
noncomputable abbrev Q51.gt := fun (answer : ℝ) => answer = ((12 : ℝ) * (π : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q51
find (answer : ℝ) s.t.
  (radius : ℝ)
  (revolutions : ℝ)
  (time : ℝ)
  (h₀ : radius = 9)
  (h₁ : revolutions = 2)
  (h₂ : time = 3)
: (answer = (2 * π * radius * revolutions) / time)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q51
find_all (answer : ℝ) iff
  (radius : ℝ)
  (revolutions : ℝ)
  (time : ℝ)
  (h₀ : radius = 9)
  (h₁ : revolutions = 2)
  (h₂ : time = 3)
: (answer = (2 * π * radius * revolutions) / time)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
