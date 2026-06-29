import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the value of $513^2 - 487^2$?

## Answer
26000
-/

-- # Formal Answer
noncomputable abbrev Q129.gt := fun (answer : ℝ) => answer = (26000 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q129
find (answer : ℝ) s.t.
  (a : ℕ)
  (b : ℕ)
  (h₀ : a = 513)
  (h₁ : b = 487)
: (a^2 - b^2 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q129
find_all (answer : ℝ) iff
  (a : ℕ)
  (b : ℕ)
  (h₀ : a = 513)
  (h₁ : b = 487)
: (a^2 - b^2 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
