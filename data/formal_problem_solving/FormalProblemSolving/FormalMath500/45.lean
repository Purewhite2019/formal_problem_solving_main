import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the positive square root of the product $10 \times 15 \times 24$?

## Answer
60
-/

-- # Formal Answer
noncomputable abbrev Q45.gt := fun (answer : ℝ) => answer = (60 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q45
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : x = Real.sqrt (10 * 15 * 24))
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q45
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : x = Real.sqrt (10 * 15 * 24))
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
