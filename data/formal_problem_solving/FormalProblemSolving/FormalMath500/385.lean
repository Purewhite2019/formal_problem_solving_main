import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x)=|x-p|+|x-15|+|x-p-15|,$ where $0 < p < 15.$ Determine the minimum value taken by $f(x)$ for $x$ in the interval $p \leq x\leq15.$

## Answer
15
-/

-- # Formal Answer
noncomputable abbrev Q385.gt := fun (answer : ℝ) => answer = (15 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q385
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (f : ℝ → ℝ)
  (p : ℝ)
  (hp : p < 15 ∧ p > 0)
  (hf : ∀ x : ℝ, f x = abs (x-p) + abs (x-15)+ abs (x-p-15))
  (hS : S = {y : ℝ | ∃ x : ℝ, x ≥ p ∧ x ≤ 15 ∧ y = f x })
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q385
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (f : ℝ → ℝ)
  (p : ℝ)
  (hp : p < 15 ∧ p > 0)
  (hf : ∀ x : ℝ, f x = abs (x-p) + abs (x-15)+ abs (x-p-15))
  (hS : S = {y : ℝ | ∃ x : ℝ, x ≥ p ∧ x ≤ 15 ∧ y = f x })
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
