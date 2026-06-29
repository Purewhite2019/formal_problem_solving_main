import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the midpoint of the line segment between $(-5,5)$ and $(3,7)$. Express your answer as an ordered pair, $(x,y)$.

## Answer
(-1,6)
-/

-- # Formal Answer
noncomputable abbrev Q214.gt := fun (answer : ℝ × ℝ) => answer = (((-1 : ℝ), (6 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q214
find (answer : ℝ × ℝ) s.t.
  (midpoint : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ)
  (hp : ∀ (p₁ p₂ : ℝ × ℝ), midpoint p₁ p₂ = ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2))
: (midpoint (-5, 5) (3, 7) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q214
find_all (answer : ℝ × ℝ) iff
  (midpoint : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ)
  (hp : ∀ (p₁ p₂ : ℝ × ℝ), midpoint p₁ p₂ = ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2))
: (midpoint (-5, 5) (3, 7) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
