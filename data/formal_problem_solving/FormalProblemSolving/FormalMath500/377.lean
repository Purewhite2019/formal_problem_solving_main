import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve
\[\sqrt{x + \sqrt{3x + 6}} + \sqrt{x - \sqrt{3x + 6}} = 6.\]

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q377.gt := fun (answer : ℝ) => answer = (10 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q377
find (answer : ℝ) s.t.
: (3 * answer + 6 ≥ 0) ∧ (answer ≥ Real.sqrt (3 * answer + 6)) ∧ (Real.sqrt (answer + Real.sqrt (3 * answer + 6)) + Real.sqrt (answer - Real.sqrt (3 * answer + 6)) = 6)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q377
find_all (answer : ℝ) iff
: (3 * answer + 6 ≥ 0) ∧ (answer ≥ Real.sqrt (3 * answer + 6)) ∧ (Real.sqrt (answer + Real.sqrt (3 * answer + 6)) + Real.sqrt (answer - Real.sqrt (3 * answer + 6)) = 6)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
