import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify $\frac{(10r^3)(4r^6)}{8r^4}$.

## Answer
5r^5
-/

-- # Formal Answer
noncomputable abbrev Q167.gt := fun (r : ℝ) (answer : ℝ) => answer = ((5 : ℝ) * (r ^ (5 : ℕ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q167
  (r : ℝ)
find (answer : ℝ) s.t.
  (h₀ : r ≠ 0)
: ((10 * r^3 * 4 * r^6) / (8 * r^4) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q167
  (r : ℝ)
find_all (answer : ℝ) iff
  (h₀ : r ≠ 0)
: ((10 * r^3 * 4 * r^6) / (8 * r^4) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
