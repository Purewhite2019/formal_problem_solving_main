import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve for $a$: $\dfrac{8^{-1}}{4^{-1}}-a^{-1}=1$.

## Answer
-2
-/

-- # Formal Answer
noncomputable abbrev Q295.gt := fun (answer : ℝ) => answer = (-2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q295
find (answer : ℝ) s.t.
  (a : ℝ)
  (h₀ : a ≠ 0)
  (h₁ : 8⁻¹ / 4⁻¹ - a⁻¹ = 1)
: (answer = a)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q295
find_all (answer : ℝ) iff
  (a : ℝ)
  (h₀ : a ≠ 0)
  (h₁ : 8⁻¹ / 4⁻¹ - a⁻¹ = 1)
: (answer = a)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
