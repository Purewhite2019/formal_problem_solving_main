import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the remainder when $1 + 2 + 3 + 4 + \dots + 9 + 10$ is divided by 9?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q272.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q272
find (answer : ℕ) s.t.
: ((∑ i in Finset.range 10, i + 1) % 9 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q272
find_all (answer : ℕ) iff
: ((∑ i in Finset.range 10, i + 1) % 9 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
