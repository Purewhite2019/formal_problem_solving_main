import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The graphs of $x^2 + y^2 + 6x - 24y + 72 = 0$ and $x^2 - y^2 + 6x + 16y - 46 = 0$ intersect at four points.  Compute the sum of the distances from these four points to the point $(-3,2).$

## Answer
40
-/

-- # Formal Answer
noncomputable abbrev Q316.gt := fun (answer : ℝ) => answer = (40 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q316
find (answer : ℝ) s.t.
  (S : Finset (ℝ × ℝ))
  (hS : S = {x : (ℝ × ℝ)| (x.1)^2 + (x.2)^2 + 6*(x.1) - 24*(x.2) + 72 = 0 ∧ (x.1)^2 - (x.2)^2 + 6*(x.1) + 16*(x.2) - 46 = 0 })
: (answer = ∑ i∈ S, Real.sqrt ((i.1+3)^2 + (i.2-2)^2))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q316
find_all (answer : ℝ) iff
  (S : Finset (ℝ × ℝ))
  (hS : S = {x : (ℝ × ℝ)| (x.1)^2 + (x.2)^2 + 6*(x.1) - 24*(x.2) + 72 = 0 ∧ (x.1)^2 - (x.2)^2 + 6*(x.1) + 16*(x.2) - 46 = 0 })
: (answer = ∑ i∈ S, Real.sqrt ((i.1+3)^2 + (i.2-2)^2))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
