import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $a$ is the $x$-intercept, $b$ is the $y$-intercept, and $m$ is the slope of the line with equation $\frac{x}4 + \frac{y}{12} = 1$, then what is the value of $a + b + m$?

## Answer
13
-/

-- # Formal Answer
noncomputable abbrev Q202.gt := fun (answer : ℝ) => answer = (13 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q202
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (m : ℝ)
  (h_a : a / 4 + (0 : ℝ) / 12 = 1)
  (h_b : (0 : ℝ) / 4 + b / 12 = 1)
  (ha : a ≠ 0)
  (h_m : m = (b - 0) / (0 - a))
: (answer = a + b + m)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q202
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (m : ℝ)
  (h_a : a / 4 + (0 : ℝ) / 12 = 1)
  (h_b : (0 : ℝ) / 4 + b / 12 = 1)
  (ha : a ≠ 0)
  (h_m : m = (b - 0) / (0 - a))
: (answer = a + b + m)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
