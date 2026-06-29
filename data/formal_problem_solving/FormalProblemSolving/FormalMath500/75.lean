import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given
\[\mathbf{A} = \begin{pmatrix} 3 & 7 \\ -1 & -2 \end{pmatrix},\]compute $\mathbf{A}^{27} + \mathbf{A}^{31} + \mathbf{A}^{40}.$

## Answer
\begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}
-/

-- # Formal Answer
noncomputable abbrev Q75.gt := fun (answer : Matrix (Fin 2) (Fin 2) ℤ) => answer =
  (Matrix.vecCons
      (Matrix.vecCons (-1 : ℤ) (Matrix.vecCons (0 : ℤ) (![] : Fin (0 : ℕ) → ℤ) : Fin (Nat.succ (0 : ℕ) : ℕ) → ℤ) :
        Fin ((Nat.succ (0 : ℕ) : ℕ).succ : ℕ) → ℤ)
      (Matrix.vecCons
          (Matrix.vecCons (0 : ℤ) (Matrix.vecCons (-1 : ℤ) (![] : Fin (0 : ℕ) → ℤ) : Fin (Nat.succ (0 : ℕ) : ℕ) → ℤ) :
            Fin (Nat.succ (1 : ℕ) : ℕ) → ℤ)
          (![] : Fin (0 : ℕ) → Fin ((Nat.succ (0 : ℕ) : ℕ).succ : ℕ) → ℤ) :
        Fin (Nat.succ (0 : ℕ) : ℕ) → Fin ((Nat.succ (0 : ℕ) : ℕ).succ : ℕ) → ℤ) :
    Fin ((Nat.succ (0 : ℕ) : ℕ).succ : ℕ) → Fin ((Nat.succ (0 : ℕ) : ℕ).succ : ℕ) → ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q75
find (answer : Matrix (Fin 2) (Fin 2) ℤ) s.t.
  (A : Matrix (Fin 2) (Fin 2) ℤ)
  (h₀ : A = ![![3, 7], ![-1, -2]])
: (A^27 + A^31 + A^40 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q75
find_all (answer : Matrix (Fin 2) (Fin 2) ℤ) iff
  (A : Matrix (Fin 2) (Fin 2) ℤ)
  (h₀ : A = ![![3, 7], ![-1, -2]])
: (A^27 + A^31 + A^40 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
