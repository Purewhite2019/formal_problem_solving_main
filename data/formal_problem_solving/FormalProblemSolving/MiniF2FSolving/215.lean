import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Expand the product $(x+1)^2 \cdot x$.

## Answer
x^3 + 2x^2 + x
-/

-- # Formal Answer
noncomputable abbrev Q215.gt := fun (x : ℝ) (answer : ℝ) => answer = (((x ^ (3 : ℕ) : ℝ) + ((2 : ℝ) * (x ^ (2 : ℕ) : ℝ) : ℝ) : ℝ) + x : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q215
  (x : ℝ)
find (answer : ℝ) s.t.
: (answer = (x + 1)^2 * x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q215
  (x : ℝ)
find_all (answer : ℝ) iff
: (answer = (x + 1)^2 * x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
