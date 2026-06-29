import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find all $p$ which satisfy both the inequalities $0\ge 54p-144$ and $0>12-20p$. Express your answer in interval notation, reducing any fractions in your answer.

## Answer
\left(\frac{3}{5},\frac{8}{3}\right]
-/

-- # Formal Answer
noncomputable abbrev Q146.gt := fun (answer : Set ℝ) => answer = (Set.Ioc (3 / 5 : ℝ) (8 / 3 : ℝ) : Set ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q146
find (answer : Set ℝ) s.t.
: (answer = {p : ℝ | 0 ≥ 54 * p - 144 ∧ 0 > 12 - 20 * p})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q146
find_all (answer : Set ℝ) iff
: (answer = {p : ℝ | 0 ≥ 54 * p - 144 ∧ 0 > 12 - 20 * p})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
