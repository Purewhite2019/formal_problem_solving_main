import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The set of vectors $\mathbf{v}$ such that
\[\operatorname{proj}_{\begin{pmatrix} 2 \\ 1 \end{pmatrix}} \mathbf{v} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}\]lie on a line.  Enter the equation of this line in the form "$y = mx + b$".

## Answer
y = -2x
-/

-- # Formal Answer
noncomputable abbrev Q84.gt := fun (answer : ℝ × ℝ) => answer = (((-2 : ℝ), (0 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q84
find (answer : ℝ × ℝ) s.t.
: (∀ (x y : ℝ), (2 * x + y = 0) ↔ (y = answer.1 * x + answer.2))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q84
find_all (answer : ℝ × ℝ) iff
: (∀ (x y : ℝ), (2 * x + y = 0) ↔ (y = answer.1 * x + answer.2))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
