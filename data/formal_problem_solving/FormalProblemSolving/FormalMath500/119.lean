import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For how many values of $x$ is the expression $\frac{x-5}{x^2-9}$  not defined?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q119.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q119
find (answer : ℕ) s.t.
  (S : Set ℝ)
  (h₀ : S =  {x:ℝ | x^2 - 9 = 0})
: (S.encard = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q119
find_all (answer : ℕ) iff
  (S : Set ℝ)
  (h₀ : S =  {x:ℝ | x^2 - 9 = 0})
: (S.encard = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
