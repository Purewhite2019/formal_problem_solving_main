import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What value of $x$ will give the minimum value for $x^2- 14x + 3$?

## Answer
7
-/

-- # Formal Answer
noncomputable abbrev Q304.gt := fun (answer : ℝ) => answer = (7 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q304
find (answer : ℝ) s.t.
: (∀ x : ℝ, answer^2 - 14 * answer + 3 ≤ x^2 - 14 * x + 3)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q304
find_all (answer : ℝ) iff
: (∀ x : ℝ, answer^2 - 14 * answer + 3 ≤ x^2 - 14 * x + 3)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
