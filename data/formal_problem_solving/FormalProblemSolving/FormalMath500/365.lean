import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $x$, $y$, and $z$ are positive with $xy = 24$, $xz = 48$, and $yz = 72$, then find $x + y + z.$

## Answer
22
-/

-- # Formal Answer
noncomputable abbrev Q365.gt := fun (answer : ℝ) => answer = (22 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q365
find (answer : ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h₀ : x > 0 ∧ y > 0 ∧ z > 0)
  (h₁ : x * y = 24)
  (h₂ : x * z = 48)
  (h₃ : y * z = 72)
: (answer = x + y + z)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q365
find_all (answer : ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h₀ : x > 0 ∧ y > 0 ∧ z > 0)
  (h₁ : x * y = 24)
  (h₂ : x * z = 48)
  (h₃ : y * z = 72)
: (answer = x + y + z)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
