import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $\mathbf{a},$ $\mathbf{b},$ $\mathbf{c}$ be three vectors such that
\[\mathbf{a} \times \mathbf{b} = \begin{pmatrix} 6 \\ -7 \\ 3 \end{pmatrix}, \quad \mathbf{a} \times \mathbf{c} = \begin{pmatrix} 4 \\ 7 \\ 2 \end{pmatrix}, \quad \mathbf{b} \times \mathbf{c} = \begin{pmatrix} 1 \\ -7 \\ 18 \end{pmatrix}.\]Compute $(2 \mathbf{b} - \mathbf{a}) \times (3 \mathbf{c} + \mathbf{a}).$

## Answer
\begin{pmatrix} -18 \\ -49 \\ 96 \end{pmatrix}
-/

-- # Formal Answer
noncomputable abbrev Q68.gt := fun (answer : ℝ × ℝ × ℝ) => answer = (((-18 : ℝ), (((-49 : ℝ), (96 : ℝ)) : ℝ × ℝ)) : ℝ × ℝ × ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q68
find (answer : ℝ × ℝ × ℝ) s.t.
  (a : ℝ × ℝ × ℝ)
  (b : ℝ × ℝ × ℝ)
  (c : ℝ × ℝ × ℝ)
  (f : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ))
  (hf : ∀ u v : (ℝ × ℝ × ℝ), f u v = (u.2.1 * v.2.2 - u.2.2 * v.2.1, u.2.2 * v.1 - u.1 * v.2.2, u.1 * v.2.1 - u.2.1 * v.1))
  (habc : f a b = (6, -7, 3) ∧ f a c = (4, 7, 2) ∧ f b c = (1, -7, 18))
: (answer = f (2*b - a) (3*c + a))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q68
find_all (answer : ℝ × ℝ × ℝ) iff
  (a : ℝ × ℝ × ℝ)
  (b : ℝ × ℝ × ℝ)
  (c : ℝ × ℝ × ℝ)
  (f : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ))
  (hf : ∀ u v : (ℝ × ℝ × ℝ), f u v = (u.2.1 * v.2.2 - u.2.2 * v.2.1, u.2.2 * v.1 - u.1 * v.2.2, u.1 * v.2.1 - u.2.1 * v.1))
  (habc : f a b = (6, -7, 3) ∧ f a c = (4, 7, 2) ∧ f b c = (1, -7, 18))
: (answer = f (2*b - a) (3*c + a))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
