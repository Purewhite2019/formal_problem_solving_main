import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Solve
\[\frac{|x^2 - 81|}{x^2 - 36x} < 0.\]

## Answer
(0,9) \cup (9,36)
-/

-- # Formal Answer
noncomputable abbrev Q350.gt := fun (answer : Set ℝ) => answer = ({x : ℝ | (0 : ℝ) < x ∧ x < (9 : ℝ) ∨ (9 : ℝ) < x ∧ x < (36 : ℝ)} : Set ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q350
find (answer : Set ℝ) s.t.
: (answer = {x : ℝ | x^2 -36*x ≠ 0 ∧ (abs (x^2 - 81)) / (x^2 - 36*x) < 0})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q350
find_all (answer : Set ℝ) iff
: (answer = {x : ℝ | x^2 -36*x ≠ 0 ∧ (abs (x^2 - 81)) / (x^2 - 36*x) < 0})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
