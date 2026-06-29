import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $x_1,$ $x_2,$ $x_3,$ $y_1,$ $y_2,$ and $y_3$ be real numbers such that
\begin{align*}
(x_1 - x_2)^2 + (y_1 - y_2)^2 &= 9, \\
(x_1 - x_3)^2 + (y_1 - y_3)^2 &= 16, \\
(x_2 - x_3)^2 + (y_2 - y_3)^2 &= 25.
\end{align*}Find $\begin{vmatrix} x_1 & y_1 & 1 \\ x_2 & y_2 & 1 \\ x_3 & y_3 & 1 \end{vmatrix}^2.$

## Answer
144
-/

-- # Formal Answer
noncomputable abbrev Q95.gt := fun (answer : ℝ) => answer = (144 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q95
find (answer : ℝ) s.t.
  (x₁ : ℝ)
  (x₂ : ℝ)
  (x₃ : ℝ)
  (y₁ : ℝ)
  (y₂ : ℝ)
  (y₃ : ℝ)
  (h₁ : (x₁ - x₂)^2 + (y₁ - y₂)^2 = 9)
  (h₂ : (x₁ - x₃)^2 + (y₁ - y₃)^2 = 16)
  (h₃ : (x₂ - x₃)^2 + (y₂ - y₃)^2 = 25)
: ((Matrix.det ![![x₁, y₁, 1], ![x₂, y₂, 1], ![x₃, y₃, 1]])^2 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q95
find_all (answer : ℝ) iff
  (x₁ : ℝ)
  (x₂ : ℝ)
  (x₃ : ℝ)
  (y₁ : ℝ)
  (y₂ : ℝ)
  (y₃ : ℝ)
  (h₁ : (x₁ - x₂)^2 + (y₁ - y₂)^2 = 9)
  (h₂ : (x₁ - x₃)^2 + (y₁ - y₃)^2 = 16)
  (h₃ : (x₂ - x₃)^2 + (y₂ - y₃)^2 = 25)
: ((Matrix.det ![![x₁, y₁, 1], ![x₂, y₂, 1], ![x₃, y₃, 1]])^2 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
