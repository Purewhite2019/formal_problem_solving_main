import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
$441+2(21)(19)+361=x$. Solve for $x$.

## Answer
1600
-/

-- # Formal Answer
noncomputable abbrev Q192.gt := fun (answer : ℝ) => answer = (1600 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q192
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : 441 + 2 * (21) * (19) + 361 = x)
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q192
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : 441 + 2 * (21) * (19) + 361 = x)
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
