import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the number of distinct positive divisors of $(30)^4$ excluding 1 and $(30)^4$.

## Answer
123
-/

-- # Formal Answer
noncomputable abbrev Q48.gt := fun (answer : ℕ) => answer = (123 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q48
find (answer : ℕ) s.t.
  (n : ℕ)
  (S : Finset ℕ)
  (h₀ : n = 30^4)
  (hS : S = {k ∈ Nat.divisors n | k ≠ 1 ∧ k ≠ n})
: (answer = Finset.card S)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q48
find_all (answer : ℕ) iff
  (n : ℕ)
  (S : Finset ℕ)
  (h₀ : n = 30^4)
  (hS : S = {k ∈ Nat.divisors n | k ≠ 1 ∧ k ≠ n})
: (answer = Finset.card S)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
