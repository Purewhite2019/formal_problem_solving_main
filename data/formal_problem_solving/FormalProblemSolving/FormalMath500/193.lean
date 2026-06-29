import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Evaluate $(1+2i)6-3i$.

## Answer
6+9i
-/

-- # Formal Answer
noncomputable abbrev Q193.gt := fun (answer : ℝ × ℝ) => answer = (((6 : ℝ), (9 : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q193
find (answer : ℝ × ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h₀ : (1 + 2 * Complex.I) * 6 - 3 * Complex.I = (a + b * Complex.I))
: ((a, b) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q193
find_all (answer : ℝ × ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h₀ : (1 + 2 * Complex.I) * 6 - 3 * Complex.I = (a + b * Complex.I))
: ((a, b) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
