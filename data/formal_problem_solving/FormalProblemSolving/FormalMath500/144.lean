import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the domain of the real-valued function $f(x)=\frac{2x-7}{\sqrt{x^2-5x+6}}$?

## Answer
(-\infty, 2) \cup (3, \infty)
-/

-- # Formal Answer
noncomputable abbrev Q144.gt := fun (answer : Set ℝ) => answer = ({x : ℝ | x < (2 : ℝ) ∨ x > (3 : ℝ)} : Set ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q144
find (answer : Set ℝ) s.t.
: (answer ={x :ℝ | x^2 - 5 * x + 6 > 0})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q144
find_all (answer : Set ℝ) iff
: (answer ={x :ℝ | x^2 - 5 * x + 6 > 0})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
