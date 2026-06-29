import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A group of $N$ students, where $N < 50$, is on a field trip. If their teacher puts them in groups of 8, the last group has 5 students. If their teacher instead puts them in groups of 6, the last group has 3 students. What is the sum of all possible values of $N$?

## Answer
66
-/

-- # Formal Answer
noncomputable abbrev Q14.gt := fun (answer : ℤ) => answer = (66 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q14
find (answer : ℤ) s.t.
  (S : Finset ℕ)
  (hS : S = Finset.filter (λ x => x % 8 = 5 ∧ x % 6 = 3) (Finset.range 50))
: (answer = ∑ k in S, k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q14
find_all (answer : ℤ) iff
  (S : Finset ℕ)
  (hS : S = Finset.filter (λ x => x % 8 = 5 ∧ x % 6 = 3) (Finset.range 50))
: (answer = ∑ k in S, k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
