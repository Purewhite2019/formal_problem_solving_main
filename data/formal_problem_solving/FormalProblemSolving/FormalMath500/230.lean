import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the sum of all of the multiples of 7 between 100 and 200?

## Answer
2107
-/

-- # Formal Answer
noncomputable abbrev Q230.gt := fun (answer : ℕ) => answer = (2107 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q230
find (answer : ℕ) s.t.
: (∑ n ∈ (Finset.filter (fun n => n % 7 = 0) (Finset.Icc 100 200)), n = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q230
find_all (answer : ℕ) iff
: (∑ n ∈ (Finset.filter (fun n => n % 7 = 0) (Finset.Icc 100 200)), n = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
