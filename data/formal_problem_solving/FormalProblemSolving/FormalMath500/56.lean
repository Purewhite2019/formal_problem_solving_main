import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $N$ be the units digit of the number $21420N$. Which nonzero value of $N$ makes this number divisible by $6$?

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q56.gt := fun (answer : ℕ) => answer = (6 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q56
find (answer : ℕ) s.t.
  (N : ℕ)
  (h_N : N ≠ 0 ∧ N ≥ 0 ∧ N ≤ 9)
  (h_divisible_by_2 : 21420 * 10 + N ≡ 0 [MOD 2])
  (h_divisible_by_3 : 21420 * 10 + N ≡ 0 [MOD 3])
: (N = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q56
find_all (answer : ℕ) iff
  (N : ℕ)
  (h_N : N ≠ 0 ∧ N ≥ 0 ∧ N ≤ 9)
  (h_divisible_by_2 : 21420 * 10 + N ≡ 0 [MOD 2])
  (h_divisible_by_3 : 21420 * 10 + N ≡ 0 [MOD 3])
: (N = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
