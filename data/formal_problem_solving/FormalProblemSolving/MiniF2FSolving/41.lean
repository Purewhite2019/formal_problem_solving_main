import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The expression $24x^2-19x-35$ can be written as $(Ax-5)(2Bx+C)$, where $A$, $B$, and $C$ are positive numbers. Find $AB-3C$.

## Answer
-9
-/

-- # Formal Answer
noncomputable abbrev Q41.gt := fun (answer : ℝ) => answer = (-9 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q41
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : ∀ x, 24 * x^2 - 19 * x - 35 = (((a * x) - 5) * ((2 * (b * x)) + c)))
: (answer = a * b - 3 * c)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q41
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : ∀ x, 24 * x^2 - 19 * x - 35 = (((a * x) - 5) * ((2 * (b * x)) + c)))
: (answer = a * b - 3 * c)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
