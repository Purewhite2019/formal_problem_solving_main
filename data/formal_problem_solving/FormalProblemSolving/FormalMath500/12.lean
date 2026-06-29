import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Sam is carrying a 2 gallon bucket of milk to his house from the barn, which is 3 miles away from his house. However, the bucket has a leak in it. For each mile he walks, there is $\frac{2}{3}$ as much milk in the bucket as at the beginning of the mile. How many gallons of milk will be in the bucket when Sam gets home?

## Answer
\frac{16}{27}
-/

-- # Formal Answer
noncomputable abbrev Q12.gt := fun (answer : ℝ) => answer = (16 / 27 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q12
find (answer : ℝ) s.t.
  (leak_ratio : ℝ)
  (h_ratio : leak_ratio = (2:ℝ) / 3)
: (answer = 2 * (leak_ratio)^3)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q12
find_all (answer : ℝ) iff
  (leak_ratio : ℝ)
  (h_ratio : leak_ratio = (2:ℝ) / 3)
: (answer = 2 * (leak_ratio)^3)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
