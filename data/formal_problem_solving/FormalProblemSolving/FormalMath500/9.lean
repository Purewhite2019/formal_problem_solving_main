import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is $\frac{9}{2}$ expressed as a decimal?

## Answer
4.5
-/

-- # Formal Answer
noncomputable abbrev Q9.gt := fun (answer : ℚ) => answer = 4.5

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q9
find (answer : ℚ) s.t.
: ((9 / 2 : ℚ) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q9
find_all (answer : ℚ) iff
: ((9 / 2 : ℚ) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
