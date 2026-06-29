import FormalProblemSolving.Basic
open BigOperators

/-
## Problem
Let $\mathbf{A}$ be a matrix such that
\[\mathbf{A} \begin{pmatrix} 3 \\ 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 2 \\ 4 \\ -3 \end{pmatrix} \quad \text{and} \quad \mathbf{A} \begin{pmatrix} -5 \\ 2 \\ 2 \end{pmatrix} = \begin{pmatrix} 0 \\ -5 \\ -5 \end{pmatrix}.\]Find $\mathbf{A} \begin{pmatrix} -13 \\ 3 \\ 4 \end{pmatrix}.$

## Answer
\begin{pmatrix} -2 \\ -14 \\ -7 \end{pmatrix}
-/

-- # Formal Answer
noncomputable abbrev Q76.gt := fun (answer : Fin 3 → ℝ) => answer = ![-2, -14, -7]

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q76
find (answer : Fin 3 → ℝ) s.t.
  (a : Fin 3 → ℝ)
  (b : Fin 3 → ℝ)
  (c : Fin 3 → ℝ)
  (u : Fin 3 → ℝ)
  (v : Fin 3 → ℝ)
  (t : Fin 3 → ℝ)
  (huvt : u = ![3, 1, 0] ∧ v = ![-5, 2, 2] ∧ t = ![-13, 3, 4])
  (hu : ∑ i, a i * u i = 2 ∧ ∑ i, b i * u i = 4 ∧ ∑ i, c i * u i = -3)
  (hv : ∑ i, a i * v i = 0 ∧ ∑ i, b i * v i = -5 ∧ ∑ i, c i * v i = -5)
: (∑ i, a i * t i = answer 0 ∧ ∑ i, b i * t i = answer 1 ∧ ∑ i, c i * t i = answer 2)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q76
find_all (answer : Fin 3 → ℝ) iff
  (a : Fin 3 → ℝ)
  (b : Fin 3 → ℝ)
  (c : Fin 3 → ℝ)
  (u : Fin 3 → ℝ)
  (v : Fin 3 → ℝ)
  (t : Fin 3 → ℝ)
  (huvt : u = ![3, 1, 0] ∧ v = ![-5, 2, 2] ∧ t = ![-13, 3, 4])
  (hu : ∑ i, a i * u i = 2 ∧ ∑ i, b i * u i = 4 ∧ ∑ i, c i * u i = -3)
  (hv : ∑ i, a i * v i = 0 ∧ ∑ i, b i * v i = -5 ∧ ∑ i, c i * v i = -5)
: (∑ i, a i * t i = answer 0 ∧ ∑ i, b i * t i = answer 1 ∧ ∑ i, c i * t i = answer 2)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
