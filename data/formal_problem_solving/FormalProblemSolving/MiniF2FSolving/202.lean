import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the value of the following expression?

$\frac{100^2-7^2}{70^2-11^2} \cdot \frac{(70-11)(70+11)}{(100-7)(100+7)}$

$\textbf{(A) } 1 \qquad \textbf{(B) } \frac{9951}{9950} \qquad \textbf{(C) } \frac{4780}{4779} \qquad \textbf{(D) } \frac{108}{107} \qquad \textbf{(E) } \frac{81}{80} $

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q202.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q202
find (answer : ℝ) s.t.
: (answer = ((100 ^ 2 - 7 ^ 2):ℝ) / (70 ^ 2 - 11 ^ 2) * ((70 - 11) * (70 + 11) / ((100 - 7) * (100 + 7))))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q202
find_all (answer : ℝ) iff
: (answer = ((100 ^ 2 - 7 ^ 2):ℝ) / (70 ^ 2 - 11 ^ 2) * ((70 - 11) * (70 + 11) / ((100 - 7) * (100 + 7))))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
