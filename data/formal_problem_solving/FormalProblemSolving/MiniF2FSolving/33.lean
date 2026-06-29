import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the remainder when $1999^{2000}$ is divided by $5?$

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q33.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q33
find (answer : ℕ) s.t.
: (answer = (1999^2000) % 5)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q33
find_all (answer : ℕ) iff
: (answer = (1999^2000) % 5)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
