import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $\displaystyle\frac{n+5}{n-3} = 2$ what is the value of $n$?

## Answer
11
-/

-- # Formal Answer
noncomputable abbrev Q182.gt := fun (answer : ℝ) => answer = (11 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q182
find (answer : ℝ) s.t.
  (n : ℝ)
  (h₀ : n ≠ 3)
  (h₁ : (n + 5) / (n - 3) = 2)
: (answer = n)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q182
find_all (answer : ℝ) iff
  (n : ℝ)
  (h₀ : n ≠ 3)
  (h₁ : (n + 5) / (n - 3) = 2)
: (answer = n)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
