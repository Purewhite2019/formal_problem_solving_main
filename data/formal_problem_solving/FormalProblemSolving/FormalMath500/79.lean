import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A reflection takes $\begin{pmatrix} 5 \\ 0 \end{pmatrix}$ to $\begin{pmatrix} 4 \\ 3 \end{pmatrix}.$  Which vector does the reflection take $\begin{pmatrix} -2 \\ 3 \end{pmatrix}$ to?

## Answer
\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}
-/

-- # Formal Answer
noncomputable abbrev Q79.gt := fun (answer : Fin 2 → ℝ) => answer =
  (Matrix.vecCons (1 / 5 : ℝ) (Matrix.vecCons (-18 / 5 : ℝ) (![] : Fin (0 : ℕ) → ℝ) : Fin (Nat.succ (0 : ℕ) : ℕ) → ℝ) :
    Fin ((Nat.succ (0 : ℕ) : ℕ).succ : ℕ) → ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q79
find (answer : Fin 2 → ℝ) s.t.
  (n : Fin 2 → ℝ)
  (h_unit : ∑ i, n i * n i = 1)
  (h_reflection : ![5,0] - 2 * ((∑ i, ![5,0] i * n i) • n) = ![4,3])
: (answer = ![-2,3] - 2 * ((∑ i, ![-2,3] i * n i) • n))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q79
find_all (answer : Fin 2 → ℝ) iff
  (n : Fin 2 → ℝ)
  (h_unit : ∑ i, n i * n i = 1)
  (h_reflection : ![5,0] - 2 * ((∑ i, ![5,0] i * n i) • n) = ![4,3])
: (answer = ![-2,3] - 2 * ((∑ i, ![-2,3] i * n i) • n))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
