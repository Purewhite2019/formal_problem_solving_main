import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is $\sqrt{53}$ in simplest radical form?

## Answer
\sqrt{53}
-/

-- # Formal Answer
noncomputable abbrev Q17.gt := fun (answer : ℝ) => answer = (√(53 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q17
find (answer : ℝ) s.t.
: (answer = Real.sqrt 53)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q17
find_all (answer : ℝ) iff
: (answer = Real.sqrt 53)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
