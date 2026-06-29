import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $f(x)=ax^4-bx^2+x+5$ and $f(-3)=2,$ then what is the value of $f(3)$?

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q222.gt := fun (answer : ℝ) => answer = (8 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q222
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (a : ℝ)
  (b : ℝ)
  (h₀ : f = fun x ↦ a * x^4 - b * x^2 + x + 5)
  (h₁ : f (-3) = 2)
: (f 3 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q222
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (a : ℝ)
  (b : ℝ)
  (h₀ : f = fun x ↦ a * x^4 - b * x^2 + x + 5)
  (h₁ : f (-3) = 2)
: (f 3 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
