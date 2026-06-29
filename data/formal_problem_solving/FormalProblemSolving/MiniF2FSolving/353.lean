import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the remainder when $1 + 2 + 2^2 + 2^3 + \dots + 2^{100}$ is divided by 7.

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q353.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q353
find (answer : ℕ) s.t.
  (S : ℕ)
  (hS : S = ∑ k in (Finset.range 101), 2^k)
: (answer = S % 7)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q353
find_all (answer : ℕ) iff
  (S : ℕ)
  (hS : S = ∑ k in (Finset.range 101), 2^k)
: (answer = S % 7)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
