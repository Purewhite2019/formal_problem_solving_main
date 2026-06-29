import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Point $A$ lies somewhere within or on the square which has opposite corners at $(0,0)$ and $(2,2)$. Point $B$ lies somewhere within or on the square which has opposite corners at points $(4,2)$ and $(5,3)$. What is the greatest possible value of the slope of the line containing points $A$ and $B$? Express your answer as a common fraction.

## Answer
\frac{3}{2}
-/

-- # Formal Answer
noncomputable abbrev Q196.gt := fun (answer : ℝ) => answer = (3 / 2 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q196
find (answer : ℝ) s.t.
: (IsGreatest {m : ℝ | ∃ (A B : ℝ × ℝ),
    A ∈ Set.Icc ((0 : ℝ), (0 : ℝ)) ((2 : ℝ), (2 : ℝ)) ∧
    B ∈ Set.Icc ((4 : ℝ), (2 : ℝ)) ((5 : ℝ), (3 : ℝ)) ∧
    m = (B.2 - A.2) / (B.1 - A.1)} answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q196
find_all (answer : ℝ) iff
: (IsGreatest {m : ℝ | ∃ (A B : ℝ × ℝ),
    A ∈ Set.Icc ((0 : ℝ), (0 : ℝ)) ((2 : ℝ), (2 : ℝ)) ∧
    B ∈ Set.Icc ((4 : ℝ), (2 : ℝ)) ((5 : ℝ), (3 : ℝ)) ∧
    m = (B.2 - A.2) / (B.1 - A.1)} answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
