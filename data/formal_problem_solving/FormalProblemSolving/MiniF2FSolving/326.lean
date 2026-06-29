import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the product of the [[real]] [[root]]s of the [[equation]] $x^2 + 18x + 30 = 2 \sqrt{x^2 + 18x + 45}$?

## Answer
20
-/

-- # Formal Answer
noncomputable abbrev Q326.gt := fun (answer : ℝ) => answer = (20 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q326
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = (x^2 + (18 * x +  30) - 2 * Real.sqrt (x^2 + (18 * x + 45))))
  (h₁ : Fintype (f⁻¹' {0}))
: (answer = ∏ x in (f⁻¹' {0}).toFinset, x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q326
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = (x^2 + (18 * x +  30) - 2 * Real.sqrt (x^2 + (18 * x + 45))))
  (h₁ : Fintype (f⁻¹' {0}))
: (answer = ∏ x in (f⁻¹' {0}).toFinset, x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
