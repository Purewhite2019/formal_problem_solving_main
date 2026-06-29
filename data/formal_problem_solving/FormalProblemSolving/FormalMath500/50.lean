import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute: $0.\overline{7}-0.\overline{4}+0.\overline{2}$. Express your answer as a common fraction.

## Answer
\frac 59
-/

-- # Formal Answer
noncomputable abbrev Q50.gt := fun (answer : ℝ) => answer = (5 / 9 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q50
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (hx : 10 * x = 7 + x)
  (hy : 10 * y = 4 + y)
  (hz : 10 * z = 2 + z)
: (answer = x - y + z)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q50
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (hx : 10 * x = 7 + x)
  (hy : 10 * y = 4 + y)
  (hz : 10 * z = 2 + z)
: (answer = x - y + z)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
