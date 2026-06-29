import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $A$ is the sum of the positive divisors of $500$, what is the sum of the distinct prime divisors of $A$?

## Answer
25
-/

-- # Formal Answer
noncomputable abbrev Q205.gt := fun (answer : ℕ) => answer = (25 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q205
find (answer : ℕ) s.t.
  (A : ℕ)
  (hA : A = ∑ k in Nat.divisors 500, k)
: (answer = ∑ k in (Nat.divisors A).filter Nat.Prime, k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q205
find_all (answer : ℕ) iff
  (A : ℕ)
  (hA : A = ∑ k in Nat.divisors 500, k)
: (answer = ∑ k in (Nat.divisors A).filter Nat.Prime, k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
