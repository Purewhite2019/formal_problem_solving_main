import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $A = (1,-11,2),$ $B = (3,-4,1),$ and $C = (-2,1,-1).$  Compute $\angle ABC,$ in degrees.

## Answer
120^\circ
-/

-- # Formal Answer
noncomputable abbrev Q77.gt := fun (answer : ℝ) => answer = (120 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q77
find (answer : ℝ) s.t.
  (BA : Fin 3 → ℝ)
  (BC : Fin 3 → ℝ)
  (hABC : BA = (![1, -11, 2] - ![3, -4, 1]) ∧ BC = (![-2, 1, -1] - ![3, -4, 1]))
: (answer = 180 * (Real.arccos ((∑ i, BA i * BC i) / Real.sqrt ((∑ i, BA i * BA i) * (∑ i, BC i * BC i)))) / Real.pi)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q77
find_all (answer : ℝ) iff
  (BA : Fin 3 → ℝ)
  (BC : Fin 3 → ℝ)
  (hABC : BA = (![1, -11, 2] - ![3, -4, 1]) ∧ BC = (![-2, 1, -1] - ![3, -4, 1]))
: (answer = 180 * (Real.arccos ((∑ i, BA i * BC i) / Real.sqrt ((∑ i, BA i * BA i) * (∑ i, BC i * BC i)))) / Real.pi)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
