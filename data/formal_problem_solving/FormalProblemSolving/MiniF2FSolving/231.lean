import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find $3x^2 y^2$ if $x$ and $y$ are [[integer]]s such that $y^2 + 3x^2 y^2 = 30x^2 + 517$.

## Answer
588
-/

-- # Formal Answer
noncomputable abbrev Q231.gt := fun (answer : ℤ) => answer = (588 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q231
find (answer : ℤ) s.t.
  (x : ℤ)
  (y : ℤ)
  (h₀ : y^2 + 3 * (x^2 * y^2) = 30 * x^2 + 517)
: (answer = 3 * (x^2 * y^2))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q231
find_all (answer : ℤ) iff
  (x : ℤ)
  (y : ℤ)
  (h₀ : y^2 + 3 * (x^2 * y^2) = 30 * x^2 + 517)
: (answer = 3 * (x^2 * y^2))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
