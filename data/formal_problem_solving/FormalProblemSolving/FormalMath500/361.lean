import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $S$ be the set of points $(a,b)$ with $0 \le a,$ $b \le 1$ such that the equation
\[x^4 + ax^3 - bx^2 + ax + 1 = 0\]has at least one real root.  Determine the area of the graph of $S.$

## Answer
\frac{1}{4}
-/

-- # Formal Answer
noncomputable abbrev Q361.gt := fun (answer : ℝ) => answer = (1 / 4 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q361
find (answer : ℝ) s.t.
: (answer = (MeasureTheory.volume { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ ∃ x : ℝ, x^4 + p.1 * x^3 - p.2 * x^2 + p.1 * x + 1 = 0 }).toReal)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q361
find_all (answer : ℝ) iff
: (answer = (MeasureTheory.volume { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ ∃ x : ℝ, x^4 + p.1 * x^3 - p.2 * x^2 + p.1 * x + 1 = 0 }).toReal)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
