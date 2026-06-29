import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A math club is having a bake sale as a fundraiser to raise money for an upcoming trip. They sell $54$ cookies at three for $\$1$, and $20$ cupcakes at $\$2$ each, and $35$ brownies at $\$1$ each. If it cost the math club $\$15$ to bake these items, what was their profit?

## Answer
78
-/

-- # Formal Answer
noncomputable abbrev Q41.gt := fun (answer : ℚ) => answer = (78 : ℚ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q41
find (answer : ℚ) s.t.
  (cookies_sold : ℕ)
  (cupcakes_sold : ℕ)
  (brownies_sold : ℕ)
  (cost : ℕ)
  (profit : ℚ)
  (h_cookies : cookies_sold = 54)
  (h_cupcakes : cupcakes_sold = 20)
  (h_brownies : brownies_sold = 35)
  (h_cost : cost = 15)
  (h_profit : profit = (cookies_sold / (3:ℚ) * 1) + (cupcakes_sold * 2) + (brownies_sold * 1) - cost)
: (profit = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q41
find_all (answer : ℚ) iff
  (cookies_sold : ℕ)
  (cupcakes_sold : ℕ)
  (brownies_sold : ℕ)
  (cost : ℕ)
  (profit : ℚ)
  (h_cookies : cookies_sold = 54)
  (h_cupcakes : cupcakes_sold = 20)
  (h_brownies : brownies_sold = 35)
  (h_cost : cost = 15)
  (h_profit : profit = (cookies_sold / (3:ℚ) * 1) + (cupcakes_sold * 2) + (brownies_sold * 1) - cost)
: (profit = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
