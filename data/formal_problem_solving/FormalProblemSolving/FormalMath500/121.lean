import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Write the expression \[\frac{4+6a}{5}-\frac{1+3a}{4}\] as a single fraction.

## Answer
\frac{11+9a}{20}
-/

-- # Formal Answer
noncomputable abbrev Q121.gt := fun (answer : ℝ → ℝ) => answer = fun (a : ℝ) => (((11 : ℝ) + ((9 : ℝ) * a : ℝ) : ℝ) / (20 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q121
find (answer : ℝ → ℝ) s.t.
: (∀ (a : ℝ), (4 + 6 * a) / 5 - (1 + 3 * a) / 4 = answer a)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q121
find_all (answer : ℝ → ℝ) iff
: (∀ (a : ℝ), (4 + 6 * a) / 5 - (1 + 3 * a) / 4 = answer a)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
