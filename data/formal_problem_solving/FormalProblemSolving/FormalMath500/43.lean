import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What integer $x$ satisfies $\frac{1}{4}<\frac{x}{7}<\frac{1}{3}$?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q43.gt := fun (answer : ℤ) => answer = (2 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q43
find (answer : ℤ) s.t.
  (x : ℤ)
  (h₀ : 1 / (4:ℚ) < x / (7:ℚ))
  (h₁ : x / (7:ℚ) < 1 / (3:ℚ))
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q43
find_all (answer : ℤ) iff
  (x : ℤ)
  (h₀ : 1 / (4:ℚ) < x / (7:ℚ))
  (h₁ : x / (7:ℚ) < 1 / (3:ℚ))
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
