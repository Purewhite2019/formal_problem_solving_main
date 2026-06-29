import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
When the two-digit integer $``\text{AB}"$ is cubed, the value is $912,\!673$. What is $A + B$?

## Answer
16
-/

-- # Formal Answer
noncomputable abbrev Q288.gt := fun (answer : ℕ) => answer = (16 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q288
find (answer : ℕ) s.t.
  (A : ℕ)
  (B : ℕ)
  (hA : A ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9])
  (hB : B ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  (h_cubed : (10 * A + B)^3 = 912673)
: (answer = A + B)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q288
find_all (answer : ℕ) iff
  (A : ℕ)
  (B : ℕ)
  (hA : A ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9])
  (hB : B ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  (h_cubed : (10 * A + B)^3 = 912673)
: (answer = A + B)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
