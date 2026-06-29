import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $\left(\sqrt[4]{11}\right)^{3x-3}=\frac{1}{5}$, what is the value of $\left(\sqrt[4]{11}\right)^{6x+2}$? Express your answer as a fraction.

## Answer
$\frac{121}{25}$
-/

-- # Formal Answer
noncomputable abbrev Q264.gt := fun (answer : ℝ) => answer = (121 / 25 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q264
find (answer : ℝ) s.t.
  (x : ℝ)
  (h : ((11:ℝ)^((1:ℝ) / 4))^(3 * x - 3) = 1 / 5)
: (answer = ((11:ℝ)^((1:ℝ) / 4))^(6 * x + 2))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q264
find_all (answer : ℝ) iff
  (x : ℝ)
  (h : ((11:ℝ)^((1:ℝ) / 4))^(3 * x - 3) = 1 / 5)
: (answer = ((11:ℝ)^((1:ℝ) / 4))^(6 * x + 2))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
