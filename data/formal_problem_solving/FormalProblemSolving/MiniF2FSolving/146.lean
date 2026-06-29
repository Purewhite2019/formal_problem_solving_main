import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x)= \sqrt{ax^2+bx} $.  For how many [[real number | real]] values of $a$ is there at least one [[positive number | positive]] value of $ b $ for which the [[domain]] of $f $ and the [[range]] of $ f $ are the same [[set]]?

$ \mathrm{(A) \ 0 } \qquad \mathrm{(B) \ 1 } \qquad \mathrm{(C) \ 2 } \qquad \mathrm{(D) \ 3 } \qquad \mathrm{(E) \ \mathrm{infinitely \ many} }  $

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q146.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q146
find (answer : ℕ) s.t.
  (S : Set ℝ)
  (hS : S = {a : ℝ | ∃ b : ℝ,  b > 0 ∧ {x : ℝ| 0 ≤ a * x^2 + b * x} = {y : ℝ | ∃ x : ℝ, 0 ≤ a * x^2 + b * x ∧ y = Real.sqrt (a * x^2 + b * x)}})
: (answer = S.encard)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q146
find_all (answer : ℕ) iff
  (S : Set ℝ)
  (hS : S = {a : ℝ | ∃ b : ℝ,  b > 0 ∧ {x : ℝ| 0 ≤ a * x^2 + b * x} = {y : ℝ | ∃ x : ℝ, 0 ≤ a * x^2 + b * x ∧ y = Real.sqrt (a * x^2 + b * x)}})
: (answer = S.encard)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
