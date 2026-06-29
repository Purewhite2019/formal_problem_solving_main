import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the [[volume]] of a [[cube]] whose [[surface area]] is twice that of a cube with volume 1?

$\mathrm{(A)}\ \sqrt{2}\qquad\mathrm{(B)}\ 2\qquad\mathrm{(C)}\ 2\sqrt{2}\qquad\mathrm{(D)}\ 4\qquad\mathrm{(E)}\ 8$

## Answer
2 * Real.sqrt 2
-/

-- # Formal Answer
noncomputable abbrev Q3.gt := fun (answer : ℝ) => answer = ((2 : ℝ) * (√(2 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q3
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (h₀ : 0 < x ∧ 0 < y)
  (h₁ : y^3 = 1)
  (h₂ : 6 * x^2 = 2 * (6 * y^2))
: (answer = x^3)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q3
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (h₀ : 0 < x ∧ 0 < y)
  (h₁ : y^3 = 1)
  (h₂ : 6 * x^2 = 2 * (6 * y^2))
: (answer = x^3)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
