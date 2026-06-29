import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $f(x)=\dfrac{2}{x+1}$, then what is the value of $f^{-1}\left(\frac{1}{5}\right)$?

## Answer
9
-/

-- # Formal Answer
noncomputable abbrev Q132.gt := fun (answer : ℝ) => answer = (9 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q132
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : f = fun x ↦ 2 / (x + 1))
: (f answer = 1/5)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q132
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : f = fun x ↦ 2 / (x + 1))
: (f answer = 1/5)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
