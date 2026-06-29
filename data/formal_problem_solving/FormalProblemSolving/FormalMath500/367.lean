import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $m$ and $n$ are integers such that $3m + 4n = 100$, what is the smallest possible value of $\left| m - n \right|$ ?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q367.gt := fun (answer : ℤ) => answer = (3 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q367
find (answer : ℤ) s.t.
  (m : ℤ)
  (n : ℤ)
  (h_eq : 3 * m + 4 * n = 100)
  (h_min : ∀ (m' n' : ℤ), 3 * m' + 4 * n' = 100 → |m - n| ≤ |m' - n'|)
: (answer = |m - n|)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q367
find_all (answer : ℤ) iff
  (m : ℤ)
  (n : ℤ)
  (h_eq : 3 * m + 4 * n = 100)
  (h_min : ∀ (m' n' : ℤ), 3 * m' + 4 * n' = 100 → |m - n| ≤ |m' - n'|)
: (answer = |m - n|)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
