import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Evaluate
\[\sin (\arcsin 0.4 + \arcsin 0.5) \cdot \sin (\arcsin 0.5 - \arcsin 0.4).\]

## Answer
\frac{9}{100}
-/

-- # Formal Answer
noncomputable abbrev Q92.gt := fun (answer : ℝ) => answer = (9 / 100 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q92
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : x = Real.arcsin 0.4)
  (h₁ : y = Real.arcsin 0.5)
: (answer = Real.sin (x + y) * Real.sin (y - x))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q92
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : x = Real.arcsin 0.4)
  (h₁ : y = Real.arcsin 0.5)
: (answer = Real.sin (x + y) * Real.sin (y - x))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
