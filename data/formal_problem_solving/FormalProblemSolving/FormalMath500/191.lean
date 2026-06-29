import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the distance, in units, between the points $(2, -6)$ and $(-4, 3)$? Express your answer in simplest radical form.

## Answer
3\sqrt{13}
-/

-- # Formal Answer
noncomputable abbrev Q191.gt := fun (answer : ℝ) => answer = ((3 : ℝ) * (√(13 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q191
find (answer : ℝ) s.t.
  (x₁ : ℝ)
  (y₁ : ℝ)
  (x₂ : ℝ)
  (y₂ : ℝ)
  (distance : ℝ)
  (h₀ : (x₁, y₁) = (2, -6))
  (h₁ : (x₂, y₂) = (-4, 3))
  (h₂ : distance = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2))
: (distance = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q191
find_all (answer : ℝ) iff
  (x₁ : ℝ)
  (y₁ : ℝ)
  (x₂ : ℝ)
  (y₂ : ℝ)
  (distance : ℝ)
  (h₀ : (x₁, y₁) = (2, -6))
  (h₁ : (x₂, y₂) = (-4, 3))
  (h₂ : distance = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2))
: (distance = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
