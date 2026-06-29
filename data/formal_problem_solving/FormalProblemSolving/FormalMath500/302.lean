import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $z$ be a complex number such that $z^5 = 1$ and $z \neq 1.$  Compute
\[z + \frac{1}{z} + z^2 + \frac{1}{z^2}.\]

## Answer
-1
-/

-- # Formal Answer
noncomputable abbrev Q302.gt := fun (answer : ℂ) => answer = (-1 : ℂ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q302
find (answer : ℂ) s.t.
  (z : ℂ)
  (h_z : z^5 = 1 ∧ z ≠ 1)
: (answer = z + 1 / z + z^2 + 1 / (z^2))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q302
find_all (answer : ℂ) iff
  (z : ℂ)
  (h_z : z^5 = 1 ∧ z ≠ 1)
: (answer = z + 1 / z + z^2 + 1 / (z^2))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
