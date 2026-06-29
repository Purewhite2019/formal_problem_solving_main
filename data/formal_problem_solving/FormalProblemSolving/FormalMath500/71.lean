import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the projection of $\mathbf{a}$ onto $\mathbf{b} = \begin{pmatrix} 2 \\ 6 \\ 3 \end{pmatrix}$ if $\mathbf{a} \cdot \mathbf{b} = 8.$

## Answer
\begin{pmatrix} 16/49 \\ 48/49 \\ 24/49 \end{pmatrix}
-/

-- # Formal Answer
noncomputable abbrev Q71.gt := fun (answer : Fin 3 → ℝ) => answer =
  (Matrix.vecCons (16 / 49 : ℝ)
      (Matrix.vecCons (48 / 49 : ℝ)
          (Matrix.vecCons (24 / 49 : ℝ) (![] : Fin (0 : ℕ) → ℝ) : Fin (Nat.succ (0 : ℕ) : ℕ) → ℝ) :
        Fin ((Nat.succ (0 : ℕ) : ℕ).succ : ℕ) → ℝ) :
    Fin (((Nat.succ (0 : ℕ) : ℕ).succ : ℕ).succ : ℕ) → ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q71
find (answer : Fin 3 → ℝ) s.t.
  (a : Fin 3 → ℝ)
  (b : Fin 3 → ℝ)
  (proj : Fin 3 → ℝ)
  (h_b : b = ![2, 6, 3])
  (h_dot : ∑ i, a i * b i = 8)
  (h_proj : ∀ i, proj i = ((∑ j, a j * b j) / (∑ j, b j * b j)) * b i)
: (proj = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q71
find_all (answer : Fin 3 → ℝ) iff
  (a : Fin 3 → ℝ)
  (b : Fin 3 → ℝ)
  (proj : Fin 3 → ℝ)
  (h_b : b = ![2, 6, 3])
  (h_dot : ∑ i, a i * b i = 8)
  (h_proj : ∀ i, proj i = ((∑ j, a j * b j) / (∑ j, b j * b j)) * b i)
: (proj = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
