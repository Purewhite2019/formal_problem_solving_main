import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $2x = 5y$ and $7y = 10z$, what is the value of $\frac{z}{x}$ expressed as a common fraction?

## Answer
$\frac{7}{25}$
-/

-- # Formal Answer
noncomputable abbrev Q199.gt := fun (answer : ℚ) => answer = (7 / 25 : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q199
find (answer : ℚ) s.t.
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : 2 * x = 5 * y)
  (h₂ : 7 * y = 10 * z)
: (answer = z / x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q199
find_all (answer : ℚ) iff
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : 2 * x = 5 * y)
  (h₂ : 7 * y = 10 * z)
: (answer = z / x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
