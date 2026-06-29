import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify
\[\frac{\sec x}{\sin x} - \frac{\sin x}{\cos x}.\]

## Answer
\cot x
-/

-- # Formal Answer
noncomputable abbrev Q73.gt := fun (x : ℝ) (answer : ℝ) => answer = ((Real.cos x : ℝ) / (Real.sin x : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q73
  (x : ℝ)
find (answer : ℝ) s.t.
: (answer = 1 / (Real.cos x * Real.sin x) - Real.sin x / Real.cos x)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q73
  (x : ℝ)
find_all (answer : ℝ) iff
: (answer = 1 / (Real.cos x * Real.sin x) - Real.sin x / Real.cos x)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
