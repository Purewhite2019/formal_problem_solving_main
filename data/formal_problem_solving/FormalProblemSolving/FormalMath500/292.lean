import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The product of integers 240 and $k$ is a perfect cube. What is the smallest possible positive value of $k$?

## Answer
900
-/

-- # Formal Answer
noncomputable abbrev Q292.gt := fun (answer : ℕ) => answer = (900 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q292
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {k : ℕ | 0 < k ∧ ∃ (n : ℕ), 240 * k = n^3})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q292
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {k : ℕ | 0 < k ∧ ∃ (n : ℕ), 240 * k = n^3})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
