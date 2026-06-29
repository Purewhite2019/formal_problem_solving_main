import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
There are an infinite number of vectors $\mathbf{v}$ such that
\[\begin{pmatrix} 1 \\ 2 \\ -5 \end{pmatrix} \times \mathbf{v} = \begin{pmatrix} 90 \\ 30 \\ 30 \end{pmatrix}.\]Find the vector $\mathbf{v}$ that has the smallest magnitude.

## Answer
\begin{pmatrix} -7 \\ 16 \\ 5 \end{pmatrix}
-/

-- # Formal Answer
noncomputable abbrev Q70.gt := fun (answer : EuclideanSpace ℝ (Fin 3)) => answer = ![(-7 : ℝ), (16 : ℝ), (5 : ℝ)]

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q70
find (answer : EuclideanSpace ℝ (Fin 3)) s.t.
  (a : EuclideanSpace ℝ (Fin 3))
  (b : EuclideanSpace ℝ (Fin 3))
  (ha : a = ![1, 2, -5])
  (hb : b = ![90, 30, 30])
: (crossProduct a answer = b) ∧ (∀ v : EuclideanSpace ℝ (Fin 3), crossProduct a v = b → ‖answer‖ ≤ ‖v‖)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q70
find_all (answer : EuclideanSpace ℝ (Fin 3)) iff
  (a : EuclideanSpace ℝ (Fin 3))
  (b : EuclideanSpace ℝ (Fin 3))
  (ha : a = ![1, 2, -5])
  (hb : b = ![90, 30, 30])
: (crossProduct a answer = b) ∧ (∀ v : EuclideanSpace ℝ (Fin 3), crossProduct a v = b → ‖answer‖ ≤ ‖v‖)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
