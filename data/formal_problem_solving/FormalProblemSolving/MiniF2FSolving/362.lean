import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x)=|x-p|+|x-15|+|x-p-15|$, where $0 < p < 15$. Determine the [[minimum]] value taken by $f(x)$ for $x$ in the [[interval]] $p \leq x\leq15$.

## Answer
15
-/

-- # Formal Answer
noncomputable abbrev Q362.gt := fun (answer : ℝ) => answer = (15 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q362
find (answer : ℝ) s.t.
  (p : ℝ)
  (hp : 0 < p)
  (hp' : p < 15)
: (IsLeast {y | ∃ x, p ≤ x ∧ x ≤ 15 ∧ y = |x - p| + |x - 15| + |x - (p + 15)|} answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q362
find_all (answer : ℝ) iff
  (p : ℝ)
  (hp : 0 < p)
  (hp' : p < 15)
: (IsLeast {y | ∃ x, p ≤ x ∧ x ≤ 15 ∧ y = |x - p| + |x - 15| + |x - (p + 15)|} answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
