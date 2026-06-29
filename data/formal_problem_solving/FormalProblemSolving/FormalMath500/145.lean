import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x) = 2x-3$ and $g(x) = x+1$. What is the value of $g(f(5)-1)$?

## Answer
7
-/

-- # Formal Answer
noncomputable abbrev Q145.gt := fun (answer : ℝ) => answer = (7 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q145
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f x = 2 * x - 3)
  (h₁ : ∀ x, g x = x + 1)
: (g (f 5 - 1) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q145
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₀ : ∀ x, f x = 2 * x - 3)
  (h₁ : ∀ x, g x = x + 1)
: (g (f 5 - 1) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
