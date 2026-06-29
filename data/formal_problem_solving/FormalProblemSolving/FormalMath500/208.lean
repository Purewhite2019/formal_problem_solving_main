import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the value of $r$ such that $$\frac{6r^2 -19r - 7}{2r-7} = 4r-3.$$

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q208.gt := fun (answer : ℝ) => answer = (4 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q208
find (answer : ℝ) s.t.
  (r : ℝ)
  (h_eq : (6 * r ^ 2 - 19 * r - 7) / (2 * r - 7) = 4 * r - 3)
: (answer = r)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q208
find_all (answer : ℝ) iff
  (r : ℝ)
  (h_eq : (6 * r ^ 2 - 19 * r - 7) / (2 * r - 7) = 4 * r - 3)
: (answer = r)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
