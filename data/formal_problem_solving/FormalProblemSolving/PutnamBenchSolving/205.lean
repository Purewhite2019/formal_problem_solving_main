import FormalProblemSolving.Basic

/-
## Problem
Show that the curve $x^3 + 3xy + y^3 = 1$ contains only one set of three distinct points, $A$, $B$, and $C$, which are vertices of an equilateral triangle, and find its area.

## Answer
3 * Real.sqrt 3 / 2
-/

-- # Formal Answer
noncomputable abbrev Q205.gt := fun (answer : ℝ) => answer = (((3 : ℝ) * (√(3 : ℝ) : ℝ) : ℝ) / (2 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q205
find (answer : ℝ) s.t.
  (curve : Set (ℝ × ℝ))
  (hcurve : curve = {c | c.1 ^ 3 + 3 * c.1 * c.2 + c.2 ^ 3 = 1})
  (equilateral : Set (ℝ × ℝ) → Prop)
  (hequilateral : ∀ S, equilateral S ↔ S.encard = 3 ∧
  ∃ d : ℝ, ∀ P ∈ S, ∀ Q ∈ S, P ≠ Q →
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = d)
: ((∃! S : Set (ℝ × ℝ), S ⊆ curve ∧ equilateral S) ∧ (∃ S : Set (ℝ × ℝ), S ⊆ curve ∧ equilateral S ∧ (MeasureTheory.volume (convexHull ℝ S)).toReal = answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q205
find_all (answer : ℝ) iff
  (curve : Set (ℝ × ℝ))
  (hcurve : curve = {c | c.1 ^ 3 + 3 * c.1 * c.2 + c.2 ^ 3 = 1})
  (equilateral : Set (ℝ × ℝ) → Prop)
  (hequilateral : ∀ S, equilateral S ↔ S.encard = 3 ∧
  ∃ d : ℝ, ∀ P ∈ S, ∀ Q ∈ S, P ≠ Q →
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = d)
: ((∃! S : Set (ℝ × ℝ), S ⊆ curve ∧ equilateral S) ∧ (∃ S : Set (ℝ × ℝ), S ⊆ curve ∧ equilateral S ∧ (MeasureTheory.volume (convexHull ℝ S)).toReal = answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
