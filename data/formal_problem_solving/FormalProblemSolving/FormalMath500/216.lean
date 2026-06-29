import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A worker receives an annual wage of $\$20{,}000$, which he always deposits into a savings account at the end of the year. By the end of the third year (when he makes the third deposit), he wants to have at least $\$66,200$ in the account to finance the purchase of a house. What is the minimal compound interest rate that the savings account must provide? Express your answer as a percentage, but do not include the percent sign.

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q216.gt := fun (answer : ℝ) => answer = (10 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q216
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {r:ℝ| r > 0 ∧ 20000 * (1 + r)^2 + 20000 * (1 + r) + 20000 ≥ 66200})
: (IsLeast S (answer / 100))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q216
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {r:ℝ| r > 0 ∧ 20000 * (1 + r)^2 + 20000 * (1 + r) + 20000 ≥ 66200})
: (IsLeast S (answer / 100))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
