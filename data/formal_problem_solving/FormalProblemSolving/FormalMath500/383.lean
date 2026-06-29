import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $\omega^{1997} = 1$ and $\omega \neq 1,$ then evaluate
\[\frac{1}{1 + \omega} + \frac{1}{1 + \omega^2} + \dots + \frac{1}{1 + \omega^{1997}}.\]

## Answer
\frac{1997}{2}
-/

-- # Formal Answer
noncomputable abbrev Q383.gt := fun (answer : ℂ) => answer = (1997 / 2 : ℂ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q383
find (answer : ℂ) s.t.
  (ω : ℂ)
  (hω : ω^1997 = 1)
  (hω_ne_one : ω ≠ 1)
: (∑ k in Finset.range 1997, 1 / (1 + ω^(k + 1)) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q383
find_all (answer : ℂ) iff
  (ω : ℂ)
  (hω : ω^1997 = 1)
  (hω_ne_one : ω ≠ 1)
: (∑ k in Finset.range 1997, 1 / (1 + ω^(k + 1)) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
