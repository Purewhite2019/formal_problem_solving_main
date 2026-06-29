import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the value of $\log_2{3} \cdot \log_3{4} \cdot \log_4{5} \cdot \log_5{6} \cdot \log_6{7} \cdot \log_7{8}$.

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q308.gt := fun (answer : ℝ) => answer = (3 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q308
find (answer : ℝ) s.t.
: (Real.logb 2 3 * Real.logb 3 4 * Real.logb 4 5 * Real.logb 5 6 * Real.logb 6 7 * Real.logb 7 8 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q308
find_all (answer : ℝ) iff
: (Real.logb 2 3 * Real.logb 3 4 * Real.logb 4 5 * Real.logb 5 6 * Real.logb 6 7 * Real.logb 7 8 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
