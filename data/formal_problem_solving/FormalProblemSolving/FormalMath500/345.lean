import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $F_1$ and $F_2$ be the foci of the ellipse $kx^2 + y^2 = 1,$ where $k > 1$ is a constant. Suppose that there is a circle which passes through $F_1$ and $F_2$ and which lies tangent to the ellipse at two points on the $x$-axis. Compute $k.$

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q345.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q345
find (answer : ℝ) s.t.
  (k : ℝ)
  (h_k : k > 1)
  (F₁ : ℝ × ℝ)
  (F₂ : ℝ × ℝ)
  (h_foci : F₁ = (0, sqrt ((k - 1) / k)) ∧ F₂ = (0, -sqrt ((k - 1) / k)))
  (h_exists : ∃ (a b r : ℝ),
      (a - F₁.1)^2 + (b - F₁.2)^2 = r^2 ∧
      (a - F₂.1)^2 + (b - F₂.2)^2 = r^2 ∧
      (a - sqrt (1 / k))^2 + (b - 0)^2 = r^2 ∧
      (a - (-sqrt (1 / k)))^2 + (b - 0)^2 = r^2)
: (answer = k)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q345
find_all (answer : ℝ) iff
  (k : ℝ)
  (h_k : k > 1)
  (F₁ : ℝ × ℝ)
  (F₂ : ℝ × ℝ)
  (h_foci : F₁ = (0, sqrt ((k - 1) / k)) ∧ F₂ = (0, -sqrt ((k - 1) / k)))
  (h_exists : ∃ (a b r : ℝ),
      (a - F₁.1)^2 + (b - F₁.2)^2 = r^2 ∧
      (a - F₂.1)^2 + (b - F₂.2)^2 = r^2 ∧
      (a - sqrt (1 / k))^2 + (b - 0)^2 = r^2 ∧
      (a - (-sqrt (1 / k)))^2 + (b - 0)^2 = r^2)
: (answer = k)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
