import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the value of $y$ in the arithmetic sequence $y + 6$, $12$, $y$?

## Answer
9
-/

-- # Formal Answer
noncomputable abbrev Q351.gt := fun (answer : ℝ) => answer = (9 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q351
find (answer : ℝ) s.t.
  (y : ℝ)
  (h₀ : y + 6 + y = 2 * 12)
: (answer = y)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q351
find_all (answer : ℝ) iff
  (y : ℝ)
  (h₀ : y + 6 + y = 2 * 12)
: (answer = y)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
