import FormalProblemSolving.Basic
open Function Nat

/-
## Problem
For any square matrix $A$, we can define $\sin A$ by the usual power series: $\sin A=\sum_{n=0}^\infty \frac{(-1)^n}{(2n+1)!}A^{2n+1}$. Prove or disprove: there exists a $2 \times 2$ matrix $A$ with real entries such that $\sin A=\begin{pmatrix} 1 & 1996 \\ 0 & 1 \end{pmatrix}$.

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q167.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q167
find (answer : Prop) s.t.
  (matsin : Matrix (Fin 2) (Fin 2) ℝ → Matrix (Fin 2) (Fin 2) ℝ)
  (mat1996 : Matrix (Fin 2) (Fin 2) ℝ)
  (hmatsin : ∀ A, matsin A = ∑' n : ℕ, ((-(1 : ℝ)) ^ n / (2 * n + 1)!) • A ^ (2 * n + 1))
  (hmat1996 : mat1996 0 0 = 1 ∧ mat1996 0 1 = 1996 ∧ mat1996 1 0 = 0 ∧ mat1996 1 1 = 1)
: ((∃ A, matsin A = mat1996) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q167
find_all (answer : Prop) iff
  (matsin : Matrix (Fin 2) (Fin 2) ℝ → Matrix (Fin 2) (Fin 2) ℝ)
  (mat1996 : Matrix (Fin 2) (Fin 2) ℝ)
  (hmatsin : ∀ A, matsin A = ∑' n : ℕ, ((-(1 : ℝ)) ^ n / (2 * n + 1)!) • A ^ (2 * n + 1))
  (hmat1996 : mat1996 0 0 = 1 ∧ mat1996 0 1 = 1996 ∧ mat1996 1 0 = 0 ∧ mat1996 1 1 = 1)
: ((∃ A, matsin A = mat1996) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
