import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $a\geq b > 1,$ what is the largest possible value of $\log_{a}(a/b) + \log_{b}(b/a)?$

$
\mathrm{(A)}\ -2      \qquad
\mathrm{(B)}\ 0     \qquad
\mathrm{(C)}\ 2      \qquad
\mathrm{(D)}\ 3      \qquad
\mathrm{(E)}\ 4
$

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q172.gt := fun (answer : ℝ) => answer = (0 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q172
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {y | ∃ (a b : ℝ), 1 < b ∧ b ≤ a ∧ y = Real.logb a (a/b) + Real.logb b (b/a)})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q172
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {y | ∃ (a b : ℝ), 1 < b ∧ b ≤ a ∧ y = Real.logb a (a/b) + Real.logb b (b/a)})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
