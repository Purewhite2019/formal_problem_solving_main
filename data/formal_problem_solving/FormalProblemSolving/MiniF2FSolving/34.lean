import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the tens digit in the sum $11^1 + 11^2 + 11^3 + \ldots + 11^9$?

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q34.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q34
find (answer : ℕ) s.t.
  (sum_powers : ℕ)
  (h_sum : sum_powers = ∑ k in (Finset.Icc 1 9), 11^k)
: (answer = (sum_powers % 100) / 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q34
find_all (answer : ℕ) iff
  (sum_powers : ℕ)
  (h_sum : sum_powers = ∑ k in (Finset.Icc 1 9), 11^k)
: (answer = (sum_powers % 100) / 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
