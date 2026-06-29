import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What real values of $x$ are not in the domain of

$f(x)=\frac{1}{|x^2+3x-4|+|x^2+9x+20|}$?

## Answer
-4
-/

-- # Formal Answer
noncomputable abbrev Q212.gt := fun (answer : ℝ) => answer = (-4 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q212
find (answer : ℝ) s.t.
  (x : ℝ)
  (hx : |x^2 + 3*x - 4| + |x^2 + 9*x + 20| = 0)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q212
find_all (answer : ℝ) iff
  (x : ℝ)
  (hx : |x^2 + 3*x - 4| + |x^2 + 9*x + 20| = 0)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
