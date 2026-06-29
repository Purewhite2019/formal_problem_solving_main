import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $x^2 - x - 1 = 0$, what is the value of $x^3 - 2x + 1$?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q134.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q134
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : x^2 - x - 1 = 0)
: (x^3 - 2 * x + 1 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q134
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : x^2 - x - 1 = 0)
: (x^3 - 2 * x + 1 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
