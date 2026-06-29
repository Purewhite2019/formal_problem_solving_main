import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify $4(3r^3+5r-6)-6(2r^3-r^2+4r)$, and express your answer in the form $Ar^2 + Br + C$, where $A$, $B$, and $C$ are integers.

## Answer
6r^2-4r-24
-/

-- # Formal Answer
noncomputable abbrev Q171.gt := fun (answer : ℤ × ℤ × ℤ) => answer = (((6 : ℤ), (((-4 : ℤ), (-24 : ℤ)) : ℤ × ℤ)) : ℤ × ℤ × ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q171
find (answer : ℤ × ℤ × ℤ) s.t.
  (r : ℤ)
  (A : ℤ)
  (B : ℤ)
  (C : ℤ)
  (h₀ : ∀ r :ℝ, 4 * (3 * r^3 + 5 * r - 6) - 6 * (2 * r^3 - r^2 + 4 * r) = A * r^2 + B * r + C)
: ((A, B, C) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q171
find_all (answer : ℤ × ℤ × ℤ) iff
  (r : ℤ)
  (A : ℤ)
  (B : ℤ)
  (C : ℤ)
  (h₀ : ∀ r :ℝ, 4 * (3 * r^3 + 5 * r - 6) - 6 * (2 * r^3 - r^2 + 4 * r) = A * r^2 + B * r + C)
: ((A, B, C) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
