import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the sum of the units digits of all the multiples of $3$ between $0$ and $50$?

## Answer
78
-/

-- # Formal Answer
noncomputable abbrev Q221.gt := fun (answer : ℕ) => answer = (78 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q221
find (answer : ℕ) s.t.
  (S : Finset ℕ)
  (hS : S = Finset.filter (λ x => 3 ∣ x) (Finset.Icc 1 49))
: (answer = ∑ k in S, (k % 10))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q221
find_all (answer : ℕ) iff
  (S : Finset ℕ)
  (hS : S = Finset.filter (λ x => 3 ∣ x) (Finset.Icc 1 49))
: (answer = ∑ k in S, (k % 10))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
