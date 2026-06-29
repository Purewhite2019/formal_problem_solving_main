import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $f(x) = \dfrac{1}{x + 2},$ what is $f(f(1))$?

## Answer
$\dfrac{3}{7}$
-/

-- # Formal Answer
noncomputable abbrev Q335.gt := fun (answer : ℝ) => answer = (3 / 7 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q335
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ x, x ≠ -2 → f x = 1 / (x + 2))
: (answer = f (f 1))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q335
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ x, x ≠ -2 → f x = 1 / (x + 2))
: (answer = f (f 1))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
