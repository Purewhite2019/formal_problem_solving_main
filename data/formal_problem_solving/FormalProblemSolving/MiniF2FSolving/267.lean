import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find $y$: $\sqrt{19+3y} = 7$.

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q267.gt := fun (answer : ℝ) => answer = (10 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q267
find (answer : ℝ) s.t.
  (y : ℝ)
  (h₀ : 0 ≤ 19 + 3 * y)
  (h₁ : Real.sqrt (19 + 3 * y) = 7)
: (answer = y)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q267
find_all (answer : ℝ) iff
  (y : ℝ)
  (h₀ : 0 ≤ 19 + 3 * y)
  (h₁ : Real.sqrt (19 + 3 * y) = 7)
: (answer = y)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
