import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What fraction of 2 feet is 3 inches? Express your answer as a common fraction.

## Answer
\frac{1}{8}
-/

-- # Formal Answer
noncomputable abbrev Q5.gt := fun (answer : ℝ) => answer = (1 / 8 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q5
find (answer : ℝ) s.t.
  (feet_to_inches : ℝ)
  (total_inches : ℝ)
  (hf : feet_to_inches = 12)
  (hf2 : total_inches = 2 * feet_to_inches)
: (answer = (3 : ℝ) / total_inches)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q5
find_all (answer : ℝ) iff
  (feet_to_inches : ℝ)
  (total_inches : ℝ)
  (hf : feet_to_inches = 12)
  (hf2 : total_inches = 2 * feet_to_inches)
: (answer = (3 : ℝ) / total_inches)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
