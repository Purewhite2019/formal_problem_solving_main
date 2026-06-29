import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
A number is selected at random from 1 through 100, inclusive.  What is the probability that the number is a multiple of 3?

## Answer
\dfrac{33}{100}
-/

-- # Formal Answer
noncomputable abbrev Q37.gt := fun (answer : ℝ) => answer = (33 / 100 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q37
find (answer : ℝ) s.t.
: (answer = (↑((Finset.Icc 1 100).filter (λ n => 3 ∣ n)).card : ℝ) / (100 : ℝ))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q37
find_all (answer : ℝ) iff
: (answer = (↑((Finset.Icc 1 100).filter (λ n => 3 ∣ n)).card : ℝ) / (100 : ℝ))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
