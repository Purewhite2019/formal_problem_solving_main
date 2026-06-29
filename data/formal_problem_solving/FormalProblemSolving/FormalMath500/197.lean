import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve for the positive value of $x$ such that $\sqrt[3]{x^2 - 4x + 4} = 16$.

## Answer
66
-/

-- # Formal Answer
noncomputable abbrev Q197.gt := fun (answer : ℝ) => answer = (66 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q197
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : x > 0)
  (h₁ : (x^2 - 4 * x + 4)^((1:ℝ)/3) = 16)
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q197
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : x > 0)
  (h₁ : (x^2 - 4 * x + 4)^((1:ℝ)/3) = 16)
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
