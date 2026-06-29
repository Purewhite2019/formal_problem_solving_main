import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The expression $10x^2-x-24$ can be written as $(Ax-8)(Bx+3),$ where $A$ and $B$ are integers. What is $AB + B$?

## Answer
12
-/

-- # Formal Answer
noncomputable abbrev Q168.gt := fun (answer : ℤ) => answer = (12 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q168
find (answer : ℤ) s.t.
  (A : ℤ)
  (B : ℤ)
  (h₀ : ∀ x, 10 * x^2 - x - 24 = (A * x - 8) * (B * x + 3))
: (A * B + B = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q168
find_all (answer : ℤ) iff
  (A : ℤ)
  (B : ℤ)
  (h₀ : ∀ x, 10 * x^2 - x - 24 = (A * x - 8) * (B * x + 3))
: (A * B + B = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
