import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
In 1992, a scoop of gelato could be purchased in Italy for 1200 lire. The same gelato would have cost $\$1.50$ in the U.S. At the equivalent exchange rate between the lire and the dollar, how many dollars would be equivalent to 1,000,000 lire?

## Answer
1250
-/

-- # Formal Answer
noncomputable abbrev Q27.gt := fun (answer : ℝ) => answer = (1250 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q27
find (answer : ℝ) s.t.
  (exchange_rate : ℝ)
  (h_exchange_rate : exchange_rate = 1.50 / 1200)
: (1000000 * exchange_rate = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q27
find_all (answer : ℝ) iff
  (exchange_rate : ℝ)
  (h_exchange_rate : exchange_rate = 1.50 / 1200)
: (1000000 * exchange_rate = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
