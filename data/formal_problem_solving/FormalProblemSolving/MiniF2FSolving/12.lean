import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find $A$ and $B$ such that
\[\frac{4x}{x^2-8x+15} = \frac{A}{x-3} + \frac{B}{x-5}\]for all $x$ besides 3 and 5. Express your answer as an ordered pair in the form $(A, B).$

## Answer
(-6, 10)
-/

-- # Formal Answer
noncomputable abbrev Q12.gt := fun (answer : ℝ × ℝ) => answer = (((-6 : ℝ), (10 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q12
find (answer : ℝ × ℝ) s.t.
  (A : ℝ)
  (B : ℝ)
  (h₀ : ∀ x, (x - 3 ≠ 0 ∧ x - 5 ≠ 0) → 4 * x / (x^2 - 8 * x + 15) = A / (x - 3) + B / (x - 5))
: (answer = (A, B))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q12
find_all (answer : ℝ × ℝ) iff
  (A : ℝ)
  (B : ℝ)
  (h₀ : ∀ x, (x - 3 ≠ 0 ∧ x - 5 ≠ 0) → 4 * x / (x^2 - 8 * x + 15) = A / (x - 3) + B / (x - 5))
: (answer = (A, B))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
