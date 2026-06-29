import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The projection of $\begin{pmatrix} 2 \\ y \\ -5 \end{pmatrix}$ onto $\begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix}$ is
\[\frac{5}{6} \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix}.\]Find $y.$

## Answer
-4
-/

-- # Formal Answer
noncomputable abbrev Q103.gt := fun (answer : ℝ) => answer = (-4 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q103
find (answer : ℝ) s.t.
  (y : ℝ)
  (h₀ : (2 * 1 + y * (-2) + (-5) * 1) / (1^2 + (-2)^2 + 1^2) = 5/6)
: (y = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q103
find_all (answer : ℝ) iff
  (y : ℝ)
  (h₀ : (2 * 1 + y * (-2) + (-5) * 1) / (1^2 + (-2)^2 + 1^2) = 5/6)
: (y = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
