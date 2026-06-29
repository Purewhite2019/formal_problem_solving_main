import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $\omega$ be a complex number such that $\omega^3 = 1.$  Find all possible values of
\[\frac{1}{1 + \omega} + \frac{1}{1 + \omega^2}.\]Enter all the possible values, separated by commas.

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q91.gt := fun (answer : ℂ) => answer = (1 : ℂ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q91
find (answer : ℂ) s.t.
  (ω : ℂ)
  (h₀ : ω^3 = 1)
: ((1 / (1 + ω)) + (1 / (1 + ω^2)) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q91
find_all (answer : ℂ) iff
  (ω : ℂ)
  (h₀ : ω^3 = 1)
: ((1 / (1 + ω)) + (1 / (1 + ω^2)) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
