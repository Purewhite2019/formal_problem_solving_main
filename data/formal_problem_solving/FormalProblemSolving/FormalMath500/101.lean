import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $\mathbf{a} = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix},$ then find the vector $\mathbf{v}$ such that $\mathbf{a} \cdot \mathbf{v} = 2$ and $\mathbf{a} \times \mathbf{v} = \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix}.$

## Answer
\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}
-/

-- # Formal Answer
noncomputable abbrev Q101.gt := fun (answer : Fin 3 → ℝ) => answer =
  (Matrix.vecCons (-1 / 3 : ℝ)
      (Matrix.vecCons (2 / 3 : ℝ)
          (Matrix.vecCons (5 / 3 : ℝ) (![] : Fin (0 : ℕ) → ℝ) : Fin (Nat.succ (0 : ℕ) : ℕ) → ℝ) :
        Fin ((Nat.succ (0 : ℕ) : ℕ).succ : ℕ) → ℝ) :
    Fin (((Nat.succ (0 : ℕ) : ℕ).succ : ℕ).succ : ℕ) → ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q101
find (answer : Fin 3 → ℝ) s.t.
  (a : Fin 3 → ℝ)
  (ha : a = ![1,1,1])
: ((∑ i, a i * answer i) = 2 ∧ (answer 2) - (answer 1) = 1 ∧ (answer 0) - (answer 2) = -2 ∧ (answer 1) - (answer 0) = 1)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q101
find_all (answer : Fin 3 → ℝ) iff
  (a : Fin 3 → ℝ)
  (ha : a = ![1,1,1])
: ((∑ i, a i * answer i) = 2 ∧ (answer 2) - (answer 1) = 1 ∧ (answer 0) - (answer 2) = -2 ∧ (answer 1) - (answer 0) = 1)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
