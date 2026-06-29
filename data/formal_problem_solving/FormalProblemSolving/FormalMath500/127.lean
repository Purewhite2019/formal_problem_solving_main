import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suzanne walks four miles every third day. What is the fewest number of miles she can walk in February?

## Answer
36
-/

-- # Formal Answer
noncomputable abbrev Q127.gt := fun (answer : ℝ) => answer = (36 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q127
find (answer : ℝ) s.t.
  (miles_per_walk : ℕ)
  (days_in_february : ℕ)
  (walks_in_february : ℕ)
  (h₀ : miles_per_walk = 4)
  (h₁ : days_in_february = 28)
  (h₂ : walks_in_february = Int.floor ((days_in_february:ℤ) / 3))
: (miles_per_walk * walks_in_february = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q127
find_all (answer : ℝ) iff
  (miles_per_walk : ℕ)
  (days_in_february : ℕ)
  (walks_in_february : ℕ)
  (h₀ : miles_per_walk = 4)
  (h₁ : days_in_february = 28)
  (h₂ : walks_in_february = Int.floor ((days_in_february:ℤ) / 3))
: (miles_per_walk * walks_in_february = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
