import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Kathy needs money for her trip to Europe. If she has 300 US dollars in the bank but wants to withdraw half of it in British pounds and half of it in euros, how many more euros than pounds will she have? Assume 1 pound = 1.64 USD and 1 euro = 1.32 USD, and round to the nearest whole number.

## Answer
22
-/

-- # Formal Answer
noncomputable abbrev Q20.gt := fun (answer : ℤ) => answer = (22 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q20
find (answer : ℤ) s.t.
  (pounds : ℝ)
  (euros : ℝ)
  (h₀ : pounds = (300 / 2) / (1.64 : ℝ))
  (h₁ : euros = (300 / 2) / (1.32 : ℝ))
: (answer = round (euros - pounds))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q20
find_all (answer : ℤ) iff
  (pounds : ℝ)
  (euros : ℝ)
  (h₀ : pounds = (300 / 2) / (1.64 : ℝ))
  (h₁ : euros = (300 / 2) / (1.32 : ℝ))
: (answer = round (euros - pounds))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
