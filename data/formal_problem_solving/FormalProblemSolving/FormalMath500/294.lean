import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A group of $N$ students, where $N < 50$, is on a field trip. If their teacher puts them in groups of 8, the last group has 5 students. If their teacher instead puts them in groups of 6, the last group has 3 students. What is the sum of all possible values of $N$?

## Answer
66
-/

-- # Formal Answer
noncomputable abbrev Q294.gt := fun (answer : ℕ) => answer = (66 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q294
find (answer : ℕ) s.t.
  (S : Finset ℕ)
  (hS : S = {n:ℕ | n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3})
: (answer = ∑ s ∈ S, s)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q294
find_all (answer : ℕ) iff
  (S : Finset ℕ)
  (hS : S = {n:ℕ | n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3})
: (answer = ∑ s ∈ S, s)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
