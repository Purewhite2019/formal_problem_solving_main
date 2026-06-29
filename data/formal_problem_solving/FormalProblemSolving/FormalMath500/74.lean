import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
One line is defined by
\[\begin{pmatrix} 3 \\ -10 \\ 1 \end{pmatrix} + t \begin{pmatrix} 2 \\ -9 \\ -2 \end{pmatrix}.\]Another line is defined by
\[\begin{pmatrix} -5 \\ -3 \\ 6 \end{pmatrix} + u \begin{pmatrix} 4 \\ -18 \\ -4 \end{pmatrix}.\]These two lines are parallel.  Find the distance between these two lines.

## Answer
7
-/

-- # Formal Answer
noncomputable abbrev Q74.gt := fun (answer : ℝ) => answer = (7 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q74
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {d : ℝ | ∃ u v : ℝ, d = Real.sqrt ((3+2*u - (-5) - 4*v)^2 + (-10-9*u -(-3)-(-18)*v)^2 + (1-2*u - 6 - (-4)*v)^2)})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q74
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {d : ℝ | ∃ u v : ℝ, d = Real.sqrt ((3+2*u - (-5) - 4*v)^2 + (-10-9*u -(-3)-(-18)*v)^2 + (1-2*u - 6 - (-4)*v)^2)})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
