import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x) = 3\sqrt{2x - 7} - 8$.  Find $f(8)$.

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q51.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q51
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = 3 * Real.sqrt (2 * x - 7) - 8)
: (answer = f 8)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q51
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = 3 * Real.sqrt (2 * x - 7) - 8)
: (answer = f 8)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
