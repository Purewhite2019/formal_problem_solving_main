import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For $0 \le x \le 40$ and $0 \le y \le 50,$ find the minimum value of
\[\sqrt{x^2 + 400} + \sqrt{y^2 + 900} + \sqrt{x^2 + y^2 - 80x - 100y + 4100}.\]

## Answer
70 \sqrt{2}
-/

-- # Formal Answer
noncomputable abbrev Q329.gt := fun (answer : ℝ) => answer = ((70 : ℝ) * (√(2 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q329
find (answer : ℝ) s.t.
: (IsLeast {s : ℝ | ∃ (x : ℝ) (y : ℝ), 0 ≤ x ∧ x ≤ 40 ∧ 0 ≤ y ∧ y ≤ 50 ∧
        s = Real.sqrt (x ^ 2 + 400) + Real.sqrt (y ^ 2 + 900) + Real.sqrt (x ^ 2 + y ^ 2 - 80 * x - 100 * y + 4100)} answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q329
find_all (answer : ℝ) iff
: (IsLeast {s : ℝ | ∃ (x : ℝ) (y : ℝ), 0 ≤ x ∧ x ≤ 40 ∧ 0 ≤ y ∧ y ≤ 50 ∧
        s = Real.sqrt (x ^ 2 + 400) + Real.sqrt (y ^ 2 + 900) + Real.sqrt (x ^ 2 + y ^ 2 - 80 * x - 100 * y + 4100)} answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
