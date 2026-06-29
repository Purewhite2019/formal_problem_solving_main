import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Which of the following is equal to $1 + \frac {1}{1 + \frac {1}{1 + 1}}$?

$\textbf{(A)}\ \frac {5}{4} \qquad \textbf{(B)}\ \frac {3}{2} \qquad \textbf{(C)}\ \frac {5}{3} \qquad \textbf{(D)}\ 2 \qquad \textbf{(E)}\ 3$

## Answer
\text{C}, which is $\frac{5}{3}$
-/

-- # Formal Answer
noncomputable abbrev Q93.gt := fun (answer : ℚ) => answer = (5 / 3 : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q93
find (answer : ℚ) s.t.
: (answer = 1 + (1 / (1 + (1 / (1 + 1)))))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q93
find_all (answer : ℚ) iff
: (answer = 1 + (1 / (1 + (1 / (1 + 1)))))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
