import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the maximum value of
\[\frac{x - y}{x^4 + y^4 + 6}\]over all real numbers $x$ and $y.$

## Answer
\frac{1}{4}
-/

-- # Formal Answer
noncomputable abbrev Q343.gt := fun (answer : ℝ) => answer = (1 / 4 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q343
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {z : ℝ | ∃ x y : ℝ, z = (x - y) / (x^4 + y^4 + 6)})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q343
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {z : ℝ | ∃ x y : ℝ, z = (x - y) / (x^4 + y^4 + 6)})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
