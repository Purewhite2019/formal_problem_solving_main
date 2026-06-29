import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
When the expression $2005^2 + 2005^0 + 2005^0 + 2005^5$ is evaluated, what are the final two digits?

## Answer
52
-/

-- # Formal Answer
noncomputable abbrev Q125.gt := fun (answer : ℕ) => answer = (52 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q125
find (answer : ℕ) s.t.
  (expression : ℕ)
  (h_expression : expression = 2005^2 + 2005^0 + 2005^0 + 2005^5)
: (answer = expression % 100)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q125
find_all (answer : ℕ) iff
  (expression : ℕ)
  (h_expression : expression = 2005^2 + 2005^0 + 2005^0 + 2005^5)
: (answer = expression % 100)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
