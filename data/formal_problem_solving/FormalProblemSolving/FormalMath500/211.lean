import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x)=\left\lfloor\left(-\frac58\right)^x\right\rfloor$ be a function that is defined for all values of $x$ in $[0,\infty)$ such that $f(x)$ is a real number. How many distinct values exist in the range of $f(x)$?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q211.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q211
find (answer : ℕ) s.t.
  (S : Set ℝ)
  (hS : S = {y:ℝ | ∃ x : ℝ, x ≥ 0 ∧ y = Int.floor ((-(5:ℝ)/ 8)^x)})
: (answer = S.encard)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q211
find_all (answer : ℕ) iff
  (S : Set ℝ)
  (hS : S = {y:ℝ | ∃ x : ℝ, x ≥ 0 ∧ y = Int.floor ((-(5:ℝ)/ 8)^x)})
: (answer = S.encard)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
