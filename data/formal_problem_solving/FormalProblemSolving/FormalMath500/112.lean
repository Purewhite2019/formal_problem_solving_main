import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The average of Amy's, Ben's, and Chris's ages is 6. Four years ago,  Chris was the same age as Amy is now. In four years, Ben's age will be $\frac{3}{5}$ of Amy's age at that time. How many years old is Chris now?

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q112.gt := fun (answer : ℕ) => answer = (10 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q112
find (answer : ℕ) s.t.
  (Amy : ℕ)
  (Ben : ℕ)
  (Chris : ℕ)
  (h₀ : (Amy + Ben + Chris) / (3:ℚ) = 6)
  (h₁ : Chris - 4 = Amy)
  (h₂ : Ben + 4 = (3:ℚ) / (5:ℚ) * (Amy + 4))
: (Chris = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q112
find_all (answer : ℕ) iff
  (Amy : ℕ)
  (Ben : ℕ)
  (Chris : ℕ)
  (h₀ : (Amy + Ben + Chris) / (3:ℚ) = 6)
  (h₁ : Chris - 4 = Amy)
  (h₂ : Ben + 4 = (3:ℚ) / (5:ℚ) * (Amy + 4))
: (Chris = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
