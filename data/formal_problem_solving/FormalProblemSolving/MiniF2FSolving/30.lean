import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $f(x)=5x-12$, find a value for $x$ so that $f^{-1}(x)=f(x+1)$.

## Answer
$\frac{47}{24}$
-/

-- # Formal Answer
noncomputable abbrev Q30.gt := fun (answer : ℝ) => answer = (47 / 24 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q30
find (answer : ℝ) s.t.
  (x : ℝ)
  (h : (x + 12) / 5 = 5 * (x + 1) - 12)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q30
find_all (answer : ℝ) iff
  (x : ℝ)
  (h : (x + 12) / 5 = 5 * (x + 1) - 12)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
