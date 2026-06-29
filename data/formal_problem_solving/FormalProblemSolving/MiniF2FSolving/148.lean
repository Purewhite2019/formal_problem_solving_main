import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify the following expression to a simplified fraction: $$\sqrt{\dfrac{\dfrac{5}{\sqrt{80}}+\dfrac{\sqrt{845}}{9}+\sqrt{45}}{\sqrt5}}.$$

## Answer
13/6
-/

-- # Formal Answer
noncomputable abbrev Q148.gt := fun (answer : ℚ) => answer = (13 / 6 : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q148
find (answer : ℚ) s.t.
  (expression : ℝ)
  (h_expression : expression = Real.sqrt ((5 / Real.sqrt 80 + Real.sqrt 845 / 9 + Real.sqrt 45) / Real.sqrt 5))
: (answer = expression)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q148
find_all (answer : ℚ) iff
  (expression : ℝ)
  (h_expression : expression = Real.sqrt ((5 / Real.sqrt 80 + Real.sqrt 845 / 9 + Real.sqrt 45) / Real.sqrt 5))
: (answer = expression)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
