import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $a+b=8$, $b+c=-3$, and $a+c= -5$, what is the value of the product $abc$?

## Answer
-120
-/

-- # Formal Answer
noncomputable abbrev Q184.gt := fun (answer : ℝ) => answer = (-120 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q184
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : a + b = 8)
  (h₁ : b + c = -3)
  (h₂ : a + c = -5)
: (a * b * c = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q184
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : a + b = 8)
  (h₁ : b + c = -3)
  (h₂ : a + c = -5)
: (a * b * c = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
