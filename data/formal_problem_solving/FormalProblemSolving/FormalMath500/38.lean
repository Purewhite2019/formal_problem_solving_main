import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Half the value of $3x-9$ is $x+37$. What is the value of $x$?

## Answer
83
-/

-- # Formal Answer
noncomputable abbrev Q38.gt := fun (answer : ℤ) => answer = (83 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q38
find (answer : ℤ) s.t.
  (x : ℤ)
  (h_equation : 3 * x - 9 = 2 * (x + 37))
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q38
find_all (answer : ℤ) iff
  (x : ℤ)
  (h_equation : 3 * x - 9 = 2 * (x + 37))
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
