import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x)=3x^4-7x^3+2x^2-bx+1$. For what value of $b$ is $f(1)=1$?

## Answer
-2
-/

-- # Formal Answer
noncomputable abbrev Q149.gt := fun (answer : ℝ) => answer = (-2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q149
find (answer : ℝ) s.t.
: (3 * (1 : ℝ) ^ 4 - 7 * (1 : ℝ) ^ 3 + 2 * (1 : ℝ) ^ 2 - answer * 1 + 1 = 1)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q149
find_all (answer : ℝ) iff
: (3 * (1 : ℝ) ^ 4 - 7 * (1 : ℝ) ^ 3 + 2 * (1 : ℝ) ^ 2 - answer * 1 + 1 = 1)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
