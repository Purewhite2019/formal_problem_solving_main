import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
An investment of $\$24,\!000$ is made in a government bond that will pay $1\%$ bi-monthly interest (meaning that the investment will increase by $1\%$ every two months). At the end of five years, what is the total number of dollars in this investment?

Express your answer to the nearest whole number.

## Answer
\$32,\!348
-/

-- # Formal Answer
noncomputable abbrev Q136.gt := fun (answer : ℕ) => answer = (32348 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q136
find (answer : ℕ) s.t.
  (initial_investment : ℝ)
  (interest_rate : ℝ)
  (time_periods : ℕ)
  (final_amount : ℝ)
  (h₀ : initial_investment = 24000)
  (h₁ : interest_rate = 0.01)
  (h₂ : time_periods = 30)
  (h₃ : final_amount = initial_investment * (1 + interest_rate) ^ time_periods)
: (round final_amount = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q136
find_all (answer : ℕ) iff
  (initial_investment : ℝ)
  (interest_rate : ℝ)
  (time_periods : ℕ)
  (final_amount : ℝ)
  (h₀ : initial_investment = 24000)
  (h₁ : interest_rate = 0.01)
  (h₂ : time_periods = 30)
  (h₃ : final_amount = initial_investment * (1 + interest_rate) ^ time_periods)
: (round final_amount = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
