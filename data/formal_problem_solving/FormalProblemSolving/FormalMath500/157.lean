import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The value of $y$ varies inversely as $\sqrt x$ and when $x=24$, $y=15$. What is $x$ when $y=3$?

## Answer
600
-/

-- # Formal Answer
noncomputable abbrev Q157.gt := fun (answer : ℝ) => answer = (600 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q157
find (answer : ℝ) s.t.
  (k : ℝ)
  (h₁ : 15 = k / Real.sqrt 24)
: (3 = k / Real.sqrt answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q157
find_all (answer : ℝ) iff
  (k : ℝ)
  (h₁ : 15 = k / Real.sqrt 24)
: (3 = k / Real.sqrt answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
