import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the largest positive integer $n$ for which there is a unique integer $k$ such that $\frac{8}{15} < \frac{n}{n + k} < \frac{7}{13}$?

## Answer
112
-/

-- # Formal Answer
noncomputable abbrev Q105.gt := fun (answer : ℕ) => answer = (112 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q105
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {n : ℕ | 0 < n ∧ ∃! k : ℕ, (8 : ℝ) / 15 < n / (n + k) ∧ (n : ℝ) / (n + k) < 7 / 13})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q105
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {n : ℕ | 0 < n ∧ ∃! k : ℕ, (8 : ℝ) / 15 < n / (n + k) ∧ (n : ℝ) / (n + k) < 7 / 13})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
