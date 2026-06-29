import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the product $CD$ of the integers $C$ and $D$ for which
\[\frac{C}{x-3}+\frac{D}{x+8}=\frac{4x-23}{x^2+5x-24}\]for all real values of $x$ except $-8$ and $3$.

## Answer
-5
-/

-- # Formal Answer
noncomputable abbrev Q341.gt := fun (answer : ℤ) => answer = (-5 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q341
find (answer : ℤ) s.t.
  (C : ℤ)
  (D : ℤ)
  (h_eq : ∀ (x : ℝ), (x ≠ -8 ∧ x ≠ 3) → C / (x - 3) + D / (x + 8) = (4 * x - 23) / (x^2 + 5 * x - 24))
: (C * D = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q341
find_all (answer : ℤ) iff
  (C : ℤ)
  (D : ℤ)
  (h_eq : ∀ (x : ℝ), (x ≠ -8 ∧ x ≠ 3) → C / (x - 3) + D / (x + 8) = (4 * x - 23) / (x^2 + 5 * x - 24))
: (C * D = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
