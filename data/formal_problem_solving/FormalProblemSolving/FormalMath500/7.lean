import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If no one shares an office, in how many ways can 3 people be assigned to 5 different offices? (Each person gets exactly one office).

## Answer
60
-/

-- # Formal Answer
noncomputable abbrev Q7.gt := fun (answer : ℕ) => answer = (60 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q7
find (answer : ℕ) s.t.
: (answer = (Nat.factorial 5) / (Nat.factorial (5-3)))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q7
find_all (answer : ℕ) iff
: (answer = (Nat.factorial 5) / (Nat.factorial (5-3)))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
