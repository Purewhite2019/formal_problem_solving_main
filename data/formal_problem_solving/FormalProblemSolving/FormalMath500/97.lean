import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A line is parameterized by
\[\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 8 \\ -1 \end{pmatrix} + t \begin{pmatrix} 2 \\ 3 \end{pmatrix}.\]The equation of the line can be expressed in the form $y = mx + b.$  Enter the ordered pair $(m,b).$

## Answer
\left( \frac{3}{2}, -13 \right)
-/

-- # Formal Answer
noncomputable abbrev Q97.gt := fun (answer : ℝ × ℝ) => answer = (((3 / 2 : ℝ), (-13 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q97
find (answer : ℝ × ℝ) s.t.
  (m : ℝ)
  (b : ℝ)
  (h : ∀ t : ℝ, -1 + 3 * t = m * (8 + 2 * t) + b)
: ((m, b) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q97
find_all (answer : ℝ × ℝ) iff
  (m : ℝ)
  (b : ℝ)
  (h : ∀ t : ℝ, -1 + 3 * t = m * (8 + 2 * t) + b)
: ((m, b) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
