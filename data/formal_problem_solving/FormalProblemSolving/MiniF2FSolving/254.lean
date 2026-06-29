import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the value of $\frac{2^{2014}+2^{2012}}{2^{2014}-2^{2012}}?$

$ \textbf{(A)}\ -1\qquad\textbf{(B)}\ 1\qquad\textbf{(C)}\ \frac{5}{3}\qquad\textbf{(D)}\ 2013\qquad\textbf{(E)}\ 2^{4024} $

## Answer
\textbf{(C)} \frac{5}{3}
-/

-- # Formal Answer
noncomputable abbrev Q254.gt := fun (answer : ℝ) => answer = (5 / 3 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q254
find (answer : ℝ) s.t.
: (answer = ((2 : ℝ)^2014 + (2 : ℝ)^2012) / ((2 : ℝ)^2014 - (2 : ℝ)^2012))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q254
find_all (answer : ℝ) iff
: (answer = ((2 : ℝ)^2014 + (2 : ℝ)^2012) / ((2 : ℝ)^2014 - (2 : ℝ)^2012))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
