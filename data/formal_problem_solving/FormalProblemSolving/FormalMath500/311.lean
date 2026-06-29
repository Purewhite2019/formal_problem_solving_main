import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $z = 1+i$ and $w = \dfrac{3z+1}{5z+7}$. Find $|w|$.

## Answer
\frac{5}{13}
-/

-- # Formal Answer
noncomputable abbrev Q311.gt := fun (answer : ℂ) => answer = ((↑(5 : ℚ) : ℂ) / (13 : ℂ) : ℂ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q311
find (answer : ℂ) s.t.
  (z : ℂ)
  (w : ℂ)
  (h_z : z = 1 + Complex.I)
  (h_w : w = (3 * z + 1) / (5 * z + 7))
: (Complex.abs w = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q311
find_all (answer : ℂ) iff
  (z : ℂ)
  (w : ℂ)
  (h_z : z = 1 + Complex.I)
  (h_w : w = (3 * z + 1) / (5 * z + 7))
: (Complex.abs w = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
