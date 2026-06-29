import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
How many elements are in the intersection of the set of all the prime numbers less than 30 and the set of all the odd numbers greater than zero?

## Answer
9
-/

-- # Formal Answer
noncomputable abbrev Q287.gt := fun (answer : ℕ) => answer = (9 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q287
find (answer : ℕ) s.t.
: (answer = (List.filter (λ n : ℕ => Nat.Prime n ∧ n < 30 ∧ n % 2 = 1) (List.range 30)).length)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q287
find_all (answer : ℕ) iff
: (answer = (List.filter (λ n : ℕ => Nat.Prime n ∧ n < 30 ∧ n % 2 = 1) (List.range 30)).length)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
