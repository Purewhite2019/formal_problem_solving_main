import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many of the first one hundred positive integers are divisible by $3, 4,$ and $5?$

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q253.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q253
find (answer : ℕ) s.t.
: (answer = ((Finset.Icc 1 100).filter (fun n => n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0)).card)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q253
find_all (answer : ℕ) iff
: (answer = ((Finset.Icc 1 100).filter (fun n => n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0)).card)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
