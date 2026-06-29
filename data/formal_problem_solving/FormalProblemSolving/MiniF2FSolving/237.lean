import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the value of$\frac{\log_2 80}{\log_{40}2}-\frac{\log_2 160}{\log_{20}2}?$$\textbf{(A) }0 \qquad \textbf{(B) }1 \qquad \textbf{(C) }\frac54 \qquad \textbf{(D) }2 \qquad \textbf{(E) }\log_2 5$

## Answer
2, which corresponds to option (D)
-/

-- # Formal Answer
noncomputable abbrev Q237.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q237
find (answer : ℝ) s.t.
: (answer = (Real.log 80 / Real.log 2) / (Real.log 2 / Real.log 40) - (Real.log 160 / Real.log 2) / (Real.log 2 / Real.log 20))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q237
find_all (answer : ℝ) iff
: (answer = (Real.log 80 / Real.log 2) / (Real.log 2 / Real.log 40) - (Real.log 160 / Real.log 2) / (Real.log 2 / Real.log 20))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
