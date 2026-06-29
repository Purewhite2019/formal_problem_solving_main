import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $t(x) = 3-g(x)$ and $g(x) = \sqrt{x}$, then what is $t(g(16))$?

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q190.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q190
find (answer : ℝ) s.t.
  (t : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, t x = 3 - g x)
  (h₁ : ∀ x, g x = Real.sqrt x)
: (t (g 16) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q190
find_all (answer : ℝ) iff
  (t : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, t x = 3 - g x)
  (h₁ : ∀ x, g x = Real.sqrt x)
: (t (g 16) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
