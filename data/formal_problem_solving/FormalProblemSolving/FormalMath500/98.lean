import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The points $O,$ $A,$ and $B$ satisfy $\overrightarrow{OA} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ and $\overrightarrow{OB} = \begin{pmatrix} -3 \\ -2 \\ 1 \end{pmatrix}.$  Find the area of triangle $OAB.$

## Answer
3 \sqrt{5}
-/

-- # Formal Answer
noncomputable abbrev Q98.gt := fun (answer : ℝ) => answer = ((3 : ℝ) * (√(5 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q98
find (answer : ℝ) s.t.
  (a : Fin 3 → ℝ)
  (b : Fin 3 → ℝ)
  (hab : a = ![1,2,3] ∧ b = ![-3,-2,1])
: (answer = Real.sqrt ((a 1 * b 2 - a 2 * b 1)^2 + (b 0 * a 2 - a 0 * b 2)^2 + (a 0 * b 1 - a 1 * b 0)^2) / 2)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q98
find_all (answer : ℝ) iff
  (a : Fin 3 → ℝ)
  (b : Fin 3 → ℝ)
  (hab : a = ![1,2,3] ∧ b = ![-3,-2,1])
: (answer = Real.sqrt ((a 1 * b 2 - a 2 * b 1)^2 + (b 0 * a 2 - a 0 * b 2)^2 + (a 0 * b 1 - a 1 * b 0)^2) / 2)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
