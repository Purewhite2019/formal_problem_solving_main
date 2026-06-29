import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The point $(a, b)$ lies on the line with the equation $3x + 2y = 12.$ When $a = 4$, what is the value of $b$?

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q185.gt := fun (answer : ℝ) => answer = (0 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q185
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h₀ : 3 * a + 2 * b = 12)
  (h₁ : a = 4)
: (b = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q185
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h₀ : 3 * a + 2 * b = 12)
  (h₁ : a = 4)
: (b = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
