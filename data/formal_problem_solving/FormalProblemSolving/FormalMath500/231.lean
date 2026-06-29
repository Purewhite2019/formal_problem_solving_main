import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the shortest distance from the point $(6, 0)$ to the line $y = 2x-2$? Express your answer in simplest radical form.

## Answer
2\sqrt{5}
-/

-- # Formal Answer
noncomputable abbrev Q231.gt := fun (answer : ℝ) => answer = ((2 : ℝ) * (√(5 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q231
find (answer : ℝ) s.t.
: (IsLeast (Set.range (λ x : ℝ => Real.sqrt ((x - 6)^2 + (2 * x - 2)^2))) answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q231
find_all (answer : ℝ) iff
: (IsLeast (Set.range (λ x : ℝ => Real.sqrt ((x - 6)^2 + (2 * x - 2)^2))) answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
