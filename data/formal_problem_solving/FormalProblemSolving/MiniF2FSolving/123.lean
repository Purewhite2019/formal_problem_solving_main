import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sum of the proper divisors of 18 is 21. What is the sum of the proper divisors of 198?

## Answer
270
-/

-- # Formal Answer
noncomputable abbrev Q123.gt := fun (answer : ℕ) => answer = (270 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q123
find (answer : ℕ) s.t.
: (answer = ∑ k in (Nat.properDivisors 198), k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q123
find_all (answer : ℕ) iff
: (answer = ∑ k in (Nat.properDivisors 198), k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
