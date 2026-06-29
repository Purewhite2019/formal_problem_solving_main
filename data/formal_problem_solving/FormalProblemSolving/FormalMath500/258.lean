import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The product of two consecutive positive even integers is 288. What is the greater of the two integers?

## Answer
18
-/

-- # Formal Answer
noncomputable abbrev Q258.gt := fun (answer : ℕ) => answer = (18 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q258
find (answer : ℕ) s.t.
: (∃ (n : ℕ), n > 0 ∧ n % 2 = 0 ∧ n * (n + 2) = 288 ∧ answer = n + 2)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q258
find_all (answer : ℕ) iff
: (∃ (n : ℕ), n > 0 ∧ n % 2 = 0 ∧ n * (n + 2) = 288 ∧ answer = n + 2)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
