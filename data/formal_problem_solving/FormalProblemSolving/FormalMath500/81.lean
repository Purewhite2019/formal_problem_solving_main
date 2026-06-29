import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A matrix $\mathbf{M}$ takes $\begin{pmatrix} 2 \\ -1 \end{pmatrix}$ to $\begin{pmatrix} 9 \\ 3 \end{pmatrix},$ and $\begin{pmatrix} 1 \\ -3 \end{pmatrix}$ to $\begin{pmatrix} 7 \\ -1 \end{pmatrix}.$  Find the image of the line $y = 2x + 1$ under $\mathbf{M}.$  Express your answer in the form "$y = mx + b$".

## Answer
y = 2x + 3
-/

-- # Formal Answer
noncomputable abbrev Q81.gt := fun (answer : ℝ × ℝ) => answer = (((2 : ℝ), (3 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q81
find (answer : ℝ × ℝ) s.t.
  (a : Fin 2 → ℝ)
  (b : Fin 2 → ℝ)
  (hab : a = ![2,-1] ∧ b = ![1,-3])
  (M1 : Fin 2 → ℝ)
  (M2 : Fin 2 → ℝ)
  (hM : ∑ i, M1 i * a i = 9 ∧ ∑ i, M2 i * a i = 3 ∧ ∑ i, M1 i * b i = 7 ∧ ∑ i, M2 i * b i = -1)
  (c : Fin 2 → ℝ)
  (d : Fin 2 → ℝ)
  (hcd : c = ![0,1] ∧ d = ![1,3])
  (e : Fin 2 → ℝ)
  (f : Fin 2 → ℝ)
  (hef : ∑ i, M1 i * c i = e 0 ∧ ∑ i, M2 i * c i = e 1 ∧ ∑ i, M1 i * d i = f 0 ∧ ∑ i, M2 i * d i = f 1)
: (answer.1 * (e 0) + answer.2 = e 1 ∧ answer.1 * (f 0) + answer.2 = f 1)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q81
find_all (answer : ℝ × ℝ) iff
  (a : Fin 2 → ℝ)
  (b : Fin 2 → ℝ)
  (hab : a = ![2,-1] ∧ b = ![1,-3])
  (M1 : Fin 2 → ℝ)
  (M2 : Fin 2 → ℝ)
  (hM : ∑ i, M1 i * a i = 9 ∧ ∑ i, M2 i * a i = 3 ∧ ∑ i, M1 i * b i = 7 ∧ ∑ i, M2 i * b i = -1)
  (c : Fin 2 → ℝ)
  (d : Fin 2 → ℝ)
  (hcd : c = ![0,1] ∧ d = ![1,3])
  (e : Fin 2 → ℝ)
  (f : Fin 2 → ℝ)
  (hef : ∑ i, M1 i * c i = e 0 ∧ ∑ i, M2 i * c i = e 1 ∧ ∑ i, M1 i * d i = f 0 ∧ ∑ i, M2 i * d i = f 1)
: (answer.1 * (e 0) + answer.2 = e 1 ∧ answer.1 * (f 0) + answer.2 = f 1)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
