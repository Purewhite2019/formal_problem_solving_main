import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The product of $3t^2+5t+a$ and $4t^2+bt-2$ is $12t^4+26t^3-8t^2-16t+6$. What is $a+b$?

## Answer
-1
-/

-- # Formal Answer
noncomputable abbrev Q149.gt := fun (answer : ℝ) => answer = (-1 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q149
find (answer : ℝ) s.t.
  (t : ℝ)
  (a : ℝ)
  (b : ℝ)
  (h₀ : ∀ t :ℝ ,(3 * t^2 + 5 * t + a) * (4 * t^2 + b * t - 2) = 12 * t^4 + 26 * t^3 - 8 * t^2 - 16 * t + 6)
: (a + b = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q149
find_all (answer : ℝ) iff
  (t : ℝ)
  (a : ℝ)
  (b : ℝ)
  (h₀ : ∀ t :ℝ ,(3 * t^2 + 5 * t + a) * (4 * t^2 + b * t - 2) = 12 * t^4 + 26 * t^3 - 8 * t^2 - 16 * t + 6)
: (a + b = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
