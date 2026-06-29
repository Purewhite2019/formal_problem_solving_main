import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Calculate $\sqrt{60x} \cdot \sqrt{12x} \cdot \sqrt{63x}$ . Express your answer in simplest radical form in terms of $x$.

Note: When entering a square root with more than one character, you must use parentheses or brackets.  For example, you should enter $\sqrt{14}$ as "sqrt(14)" or "sqrt{14}".

## Answer
36x \sqrt{35x}
-/

-- # Formal Answer
noncomputable abbrev Q315.gt := fun (x : ℝ) (answer : ℝ) => answer = (((36 : ℝ) * x : ℝ) * (√((35 : ℝ) * x : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q315
  (x : ℝ)
find (answer : ℝ) s.t.
  (hx : 0 ≤ x)
: (answer = Real.sqrt (60 * x) * Real.sqrt (12 * x) * Real.sqrt (63 * x))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q315
  (x : ℝ)
find_all (answer : ℝ) iff
  (hx : 0 ≤ x)
: (answer = Real.sqrt (60 * x) * Real.sqrt (12 * x) * Real.sqrt (63 * x))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
