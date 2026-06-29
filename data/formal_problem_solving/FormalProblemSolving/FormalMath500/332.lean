import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
At what value of $y$ is there a horizontal asymptote for the graph of the equation $y=\frac{4x^3+2x-4}{3x^3-2x^2+5x-1}$?

## Answer
\frac43
-/

-- # Formal Answer
noncomputable abbrev Q332.gt := fun (answer : ℝ) => answer = ((↑(4 : ℚ) : ℝ) / (3 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q332
find (answer : ℝ) s.t.
: (Filter.Tendsto (fun x : ℝ ↦ ((4 * x^3 + 2 * x - 4) / (3 * x^3 - 2 * x^2 + 5 * x - 1))) Filter.atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q332
find_all (answer : ℝ) iff
: (Filter.Tendsto (fun x : ℝ ↦ ((4 * x^3 + 2 * x - 4) / (3 * x^3 - 2 * x^2 + 5 * x - 1))) Filter.atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
