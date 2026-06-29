import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A sequence $(a_n)$ is defined as follows:
\[a_{i + 1} = \frac{1}{1 - a_i}\]for $i \ge 1.$  If $a_3 = a_1,$ compute $(a_9)^9.$

## Answer
-1
-/

-- # Formal Answer
noncomputable abbrev Q334.gt := fun (answer : ℂ) => answer = (-1 : ℂ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q334
find (answer : ℂ) s.t.
  (a : ℕ → ℂ)
  (h_recurrence : ∀ i, 1 ≤ i → a (i + 1) = (1 : ℂ) / (1 - a i))
  (h_denom : ∀ i, 1 ≤ i → 1 - a i ≠ 0)
  (h1 : a 3 = a 1)
: ((a 9)^9 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q334
find_all (answer : ℂ) iff
  (a : ℕ → ℂ)
  (h_recurrence : ∀ i, 1 ≤ i → a (i + 1) = (1 : ℂ) / (1 - a i))
  (h_denom : ∀ i, 1 ≤ i → 1 - a i ≠ 0)
  (h1 : a 3 = a 1)
: ((a 9)^9 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
