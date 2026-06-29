import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $\arccos x + \arccos 2x + \arccos 3x = \pi,$ then $x$ satisfies a cubic polynomial of the form
\[ax^3 + bx^2 + cx + d = 0,\]where $a,$ $b,$ $c,$ and $d$ are integers, and $a \neq 0.$  Find the smallest possible value of $|a| + |b| + |c| + |d|.$

## Answer
27
-/

-- # Formal Answer
noncomputable abbrev Q110.gt := fun (answer : ℝ) => answer = (27 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q110
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : Real.arccos x + Real.arccos (2 * x) + Real.arccos (3 * x) = π)
  (h₁ : ∃ (a b c d : ℤ), a ≠ 0 ∧ a * x^3 + b * x^2 + c * x + d = 0)
: ((∀ (a b c d : ℤ), a ≠ 0 ∧ a * x^3 + b * x^2 + c * x + d = 0 → |a| + |b| + |c| + |d| ≥ answer) ∧
  (∃ (a b c d : ℤ), a ≠ 0 ∧ a * x^3 + b * x^2 + c * x + d = 0 ∧ |a| + |b| + |c| + |d| = answer))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q110
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : Real.arccos x + Real.arccos (2 * x) + Real.arccos (3 * x) = π)
  (h₁ : ∃ (a b c d : ℤ), a ≠ 0 ∧ a * x^3 + b * x^2 + c * x + d = 0)
: ((∀ (a b c d : ℤ), a ≠ 0 ∧ a * x^3 + b * x^2 + c * x + d = 0 → |a| + |b| + |c| + |d| ≥ answer) ∧
  (∃ (a b c d : ℤ), a ≠ 0 ∧ a * x^3 + b * x^2 + c * x + d = 0 ∧ |a| + |b| + |c| + |d| = answer))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
