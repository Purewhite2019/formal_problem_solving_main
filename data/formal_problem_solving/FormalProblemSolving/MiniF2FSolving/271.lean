import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Determine all real numbers $x$ which satisfy the inequality:

<center>
$\sqrt{\sqrt{3-x}-\sqrt{x+1}}>\dfrac{1}{2}$
</center>

## Answer
\left[ ~ -1,\quad 1-\dfrac{\sqrt{127}}{32} ~ \right)
-/

-- # Formal Answer
noncomputable abbrev Q271.gt := fun (answer : Set ℝ) => answer = ({x : ℝ | (-1 : ℝ) ≤ x ∧ x < ((1 : ℝ) - ((√(127 : ℝ) : ℝ) / (32 : ℝ) : ℝ) : ℝ)} : Set ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q271
find (answer : Set ℝ) s.t.
: (answer = {x : ℝ | 0 ≤ 3 - x ∧ 0 ≤ x + 1 ∧ 1 / 2 < Real.sqrt (Real.sqrt (3 - x) - Real.sqrt (x + 1))})
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q271
find_all (answer : Set ℝ) iff
: (answer = {x : ℝ | 0 ≤ 3 - x ∧ 0 ≤ x + 1 ∧ 1 / 2 < Real.sqrt (Real.sqrt (3 - x) - Real.sqrt (x + 1))})
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
