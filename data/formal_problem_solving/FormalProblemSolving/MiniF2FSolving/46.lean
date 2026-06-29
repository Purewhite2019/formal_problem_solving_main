import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify $\left( \frac{4}{x} \right)^{-1} \left( \frac{3x^3}{x} \right)^2 \left( \frac{1}{2x} \right)^{-3}$.

## Answer
18x^8
-/

-- # Formal Answer
noncomputable abbrev Q46.gt := fun (x : ℝ) (answer : ℝ) => answer = ((18 : ℝ) * (x ^ (8 : ℕ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q46
  (x : ℝ)
find (answer : ℝ) s.t.
  (h₀ : x ≠ 0)
: (answer = (4 / x)⁻¹ * ((3 * x^3) / x)^2 * ((1 / (2 * x))⁻¹)^3)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q46
  (x : ℝ)
find_all (answer : ℝ) iff
  (h₀ : x ≠ 0)
: (answer = (4 / x)⁻¹ * ((3 * x^3) / x)^2 * ((1 / (2 * x))⁻¹)^3)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
