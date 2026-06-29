import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find $k$, if ${(3^k)}^6=3^6$.

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q143.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q143
find (answer : ℝ) s.t.
: (((3:ℝ)^answer)^6 = 3^6)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q143
find_all (answer : ℝ) iff
: (((3:ℝ)^answer)^6 = 3^6)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
