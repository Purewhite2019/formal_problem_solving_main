import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the sum of the values of $a$ that satisfy the equation $$(3)5^2-4(5-a)^2 \div 3=63?$$

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q177.gt := fun (answer : ℝ) => answer = (10 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q177
find (answer : ℝ) s.t.
  (h_fin : Set.Finite {a : ℝ | 3 * 5^2 - 4 * (5 - a)^2 / 3 = 63})
: (answer = ∑ a in h_fin.toFinset, a)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q177
find_all (answer : ℝ) iff
  (h_fin : Set.Finite {a : ℝ | 3 * 5^2 - 4 * (5 - a)^2 / 3 = 63})
: (answer = ∑ a in h_fin.toFinset, a)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
