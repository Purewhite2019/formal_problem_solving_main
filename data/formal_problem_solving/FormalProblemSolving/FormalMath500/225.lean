import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
At 50 miles per hour, how far would a car travel in $2\frac{3}{4}$ hours? Express your answer as a mixed number.

## Answer
137 \frac{1}{2}
-/

-- # Formal Answer
noncomputable abbrev Q225.gt := fun (answer : ℝ) => answer = ((↑(275 : ℚ) : ℝ) / (2 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q225
find (answer : ℝ) s.t.
  (speed : ℝ)
  (time : ℝ)
  (distance : ℝ)
  (h₀ : speed = 50)
  (h₁ : time = 2 + (3:ℚ)/4)
  (h₂ : distance = speed * time)
: (distance = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q225
find_all (answer : ℝ) iff
  (speed : ℝ)
  (time : ℝ)
  (distance : ℝ)
  (h₀ : speed = 50)
  (h₁ : time = 2 + (3:ℚ)/4)
  (h₂ : distance = speed * time)
: (distance = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
