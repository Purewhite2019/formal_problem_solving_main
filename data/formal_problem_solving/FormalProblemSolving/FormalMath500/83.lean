import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute $\sin^3 18^\circ + \sin^2 18^\circ.$

## Answer
\frac{1}{8}
-/

-- # Formal Answer
noncomputable abbrev Q83.gt := fun (answer : ℝ) => answer = (1 / 8 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q83
find (answer : ℝ) s.t.
: (answer = (Real.sin (18*Real.pi/180))^3 + (Real.sin (18*Real.pi/180))^2)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q83
find_all (answer : ℝ) iff
: (answer = (Real.sin (18*Real.pi/180))^3 + (Real.sin (18*Real.pi/180))^2)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
