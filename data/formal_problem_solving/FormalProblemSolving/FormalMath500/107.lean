import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $\mathbf{a}$ and $\mathbf{b}$ are vectors such that $\|\mathbf{a}\| = 2,$ $\|\mathbf{b}\| = 7,$ and
\[\mathbf{a} \times \mathbf{b} = \begin{pmatrix} 3 \\ 2 \\ 6 \end{pmatrix},\]then find the smallest possible angle between $\mathbf{a}$ and $\mathbf{b},$ in degrees.

## Answer
30^\circ
-/

-- # Formal Answer
noncomputable abbrev Q107.gt := fun (answer : ℝ) => answer = (30 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q107
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {x : ℝ | x > 0 ∧ 2*7*Real.sin x = Real.sqrt (3*3+2*2+6*6)})
: (IsLeast S (answer*Real.pi / 180))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q107
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {x : ℝ | x > 0 ∧ 2*7*Real.sin x = Real.sqrt (3*3+2*2+6*6)})
: (IsLeast S (answer*Real.pi / 180))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
