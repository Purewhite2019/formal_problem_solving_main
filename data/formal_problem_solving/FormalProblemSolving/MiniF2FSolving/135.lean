import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find $(\log_2 x)^2$ if $\log_2 (\log_8 x) = \log_8 (\log_2 x)$.

## Answer
27
-/

-- # Formal Answer
noncomputable abbrev Q135.gt := fun (answer : ℝ) => answer = (27 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q135
find (answer : ℝ) s.t.
  (x : ℝ)
  (hx : 1 < x)
  (h_eq : Real.logb 2 (Real.logb 8 x) = Real.logb 8 (Real.logb 2 x))
: (answer = (Real.logb 2 x)^2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q135
find_all (answer : ℝ) iff
  (x : ℝ)
  (hx : 1 < x)
  (h_eq : Real.logb 2 (Real.logb 8 x) = Real.logb 8 (Real.logb 2 x))
: (answer = (Real.logb 2 x)^2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
