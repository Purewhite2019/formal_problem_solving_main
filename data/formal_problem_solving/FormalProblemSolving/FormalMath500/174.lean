import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve the following equation for $x$: \[ \ \frac{2}{3} = \frac{4}{x-5}.\]

## Answer
11
-/

-- # Formal Answer
noncomputable abbrev Q174.gt := fun (answer : ℝ) => answer = (11 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q174
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : (2 : ℝ) / 3 = (4 : ℝ) / (x - 5))
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q174
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : (2 : ℝ) / 3 = (4 : ℝ) / (x - 5))
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
