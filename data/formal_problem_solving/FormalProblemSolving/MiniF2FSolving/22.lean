import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the remainder when $1 + 2 + 3 + 4 + \dots + 9 + 10$ is divided by 9?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q22.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q22
find (answer : ℕ) s.t.
  (sum : ℕ)
  (h_sum : sum = ∑ k in (Finset.range 11), k)
: (answer = sum % 9)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q22
find_all (answer : ℕ) iff
  (sum : ℕ)
  (h_sum : sum = ∑ k in (Finset.range 11), k)
: (answer = sum % 9)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
