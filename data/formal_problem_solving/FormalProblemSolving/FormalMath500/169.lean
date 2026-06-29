import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The solution to $-4 < 2(x - 1) < 8$ is expressed in the form $a < x < b$. Find the value of $a + b$.

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q169.gt := fun (answer : ℝ) => answer = (4 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q169
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (hab : ∀ x:ℝ, (a < x ∧ x < b) → (-4 < 2 * (x - 1) ∧ 2 * (x - 1) < 8))
  (hab2 : ∀ x:ℝ, (-4 < 2 * (x - 1) ∧ 2 * (x - 1) < 8) → (a < x ∧ x < b))
: (a + b = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q169
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (hab : ∀ x:ℝ, (a < x ∧ x < b) → (-4 < 2 * (x - 1) ∧ 2 * (x - 1) < 8))
  (hab2 : ∀ x:ℝ, (-4 < 2 * (x - 1) ∧ 2 * (x - 1) < 8) → (a < x ∧ x < b))
: (a + b = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
