import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Expand $(x+3)(2x-6)$.

## Answer
2x^2 - 18
-/

-- # Formal Answer
noncomputable abbrev Q273.gt := fun (x : ℝ) (answer : ℝ) => answer = (((2 : ℝ) * (x ^ (2 : ℕ) : ℝ) : ℝ) - (18 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q273
  (x : ℝ)
find (answer : ℝ) s.t.
: (answer = (x + 3) * (2 * x - 6))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q273
  (x : ℝ)
find_all (answer : ℝ) iff
: (answer = (x + 3) * (2 * x - 6))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
