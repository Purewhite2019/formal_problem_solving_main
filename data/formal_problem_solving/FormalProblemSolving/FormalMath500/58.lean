import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A shirt's sale price is marked $\$14.40$, which is $60\%$ off the original price. How many dollars was the original price of the shirt?

## Answer
\$36
-/

-- # Formal Answer
noncomputable abbrev Q58.gt := fun (answer : ℝ) => answer = (36 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q58
find (answer : ℝ) s.t.
  (sale_price : ℝ)
  (h₀ : sale_price = 14.40)
: (sale_price = answer * (1 - 0.60))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q58
find_all (answer : ℝ) iff
  (sale_price : ℝ)
  (h₀ : sale_price = 14.40)
: (sale_price = answer * (1 - 0.60))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
