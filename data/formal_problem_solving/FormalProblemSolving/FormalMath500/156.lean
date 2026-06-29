import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The quadratic $x^2+(2.6)x+3.6$ can be written in the form $(x+b)^2+c$, where $b$ and $c$ are constants. What is $b+c$ (as a decimal)?

## Answer
3.21
-/

-- # Formal Answer
noncomputable abbrev Q156.gt := fun (answer : ℝ) => answer = 3.21

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q156
find (answer : ℝ) s.t.
  (x : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : ∀ x : ℝ, x^2 + 2.6 * x + 3.6 = (x + b)^2 + c)
: (b + c = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q156
find_all (answer : ℝ) iff
  (x : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : ∀ x : ℝ, x^2 + 2.6 * x + 3.6 = (x + b)^2 + c)
: (b + c = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
