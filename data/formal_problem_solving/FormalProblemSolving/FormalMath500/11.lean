import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify $\sqrt{242}$.

## Answer
11\sqrt2
-/

-- # Formal Answer
noncomputable abbrev Q11.gt := fun (answer : ℝ) => answer = ((11 : ℝ) * (√(2 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q11
find (answer : ℝ) s.t.
: (answer = Real.sqrt 242)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q11
find_all (answer : ℝ) iff
: (answer = Real.sqrt 242)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
