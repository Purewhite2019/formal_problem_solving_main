import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The operation $\&$ is defined for positive integers $a$ and $b$ as $a \& b = \displaystyle\frac{\sqrt{a b + a}}{\sqrt{a b - b}}$. What is the value of $9 \& 2$? Express your answer as a common fraction in simplest radical form.

## Answer
\frac{3\sqrt{3}}{4}
-/

-- # Formal Answer
noncomputable abbrev Q164.gt := fun (answer : ℝ) => answer = (((3 : ℝ) * (√(3 : ℝ) : ℝ) : ℝ) / (4 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q164
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (hab : a = 9 ∧ b = 2)
: (answer = Real.sqrt (a * b + a) / (Real.sqrt (a * b - b)))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q164
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (hab : a = 9 ∧ b = 2)
: (answer = Real.sqrt (a * b + a) / (Real.sqrt (a * b - b)))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
