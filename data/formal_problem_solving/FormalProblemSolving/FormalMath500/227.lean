import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $\frac{\sqrt{2x}}{\sqrt{3x-1}}=\frac32$, solve for $x$. Express your answer in simplest fractional form.

## Answer
\frac9{19}
-/

-- # Formal Answer
noncomputable abbrev Q227.gt := fun (answer : ℝ) => answer = ((↑(9 : ℚ) : ℝ) / (19 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q227
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : x > (1:ℚ)/3)
  (h₁ : (sqrt (2 * x)) / (sqrt (3 * x - 1)) = (3:ℚ) / 2)
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q227
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : x > (1:ℚ)/3)
  (h₁ : (sqrt (2 * x)) / (sqrt (3 * x - 1)) = (3:ℚ) / 2)
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
