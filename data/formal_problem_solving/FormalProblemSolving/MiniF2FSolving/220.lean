import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the remainder when $2000+2001+2002+2003+2004+2005+2006$ is divided by $7$?

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q220.gt := fun (answer : ℕ) => answer = (0 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q220
find (answer : ℕ) s.t.
  (sum : ℕ)
  (h_sum : sum = 2000 + 2001 + 2002 + 2003 + 2004 + 2005 + 2006)
: (answer = sum % 7)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q220
find_all (answer : ℕ) iff
  (sum : ℕ)
  (h_sum : sum = 2000 + 2001 + 2002 + 2003 + 2004 + 2005 + 2006)
: (answer = sum % 7)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
