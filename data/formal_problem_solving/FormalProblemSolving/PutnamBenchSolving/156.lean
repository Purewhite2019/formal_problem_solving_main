import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
For which real numbers $c$ is there a straight line that intersects the curve $x^4+9x^3+cx^2+9x+4$ in four distinct points?

## Answer
{c : ℝ | c < 243 / 8}
-/

-- # Formal Answer
noncomputable abbrev Q156.gt := fun (c : ℝ) (answer : Set ℝ) => answer = ({c : ℝ | c < (243 / 8 : ℝ)} : Set ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q156
  (c : ℝ)
find (answer : Set ℝ) s.t.
: (answer = {c : ℝ | (∃ m b : ℝ,
    {x : ℝ | m * x + b = x ^ 4 + 9 * x ^ 3 + c * x ^ 2 + 9 * x + 4}.encard = 4)})
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q156
  (c : ℝ)
find_all (answer : Set ℝ) iff
: (answer = {c : ℝ | (∃ m b : ℝ,
    {x : ℝ | m * x + b = x ^ 4 + 9 * x ^ 3 + c * x ^ 2 + 9 * x + 4}.encard = 4)})
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
