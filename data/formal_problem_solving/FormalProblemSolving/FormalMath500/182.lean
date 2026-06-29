import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Hillary has eleven coins, all dimes and nickels. In total, the coins are worth 75 cents. How many nickels does she have?

## Answer
7
-/

-- # Formal Answer
noncomputable abbrev Q182.gt := fun (answer : ℕ) => answer = (7 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q182
find (answer : ℕ) s.t.
  (dimes : ℕ)
  (nickels : ℕ)
  (h₀ : dimes + nickels = 11)
  (h₁ : 10 * dimes + 5 * nickels = 75)
: (nickels = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q182
find_all (answer : ℕ) iff
  (dimes : ℕ)
  (nickels : ℕ)
  (h₀ : dimes + nickels = 11)
  (h₁ : 10 * dimes + 5 * nickels = 75)
: (nickels = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
