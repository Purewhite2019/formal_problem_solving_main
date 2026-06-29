import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $A, M,$ and $C$ be [[nonnegative integer]]s such that $A + M + C=12$. What is the maximum value of $A \cdot M \cdot C + A \cdot M + M \cdot C + A \cdot C$?

$ \mathrm{(A) \ 62 } \qquad \mathrm{(B) \ 72 } \qquad \mathrm{(C) \ 92 } \qquad \mathrm{(D) \ 102 } \qquad \mathrm{(E) \ 112 }  $

## Answer
112
-/

-- # Formal Answer
noncomputable abbrev Q262.gt := fun (answer : ℕ) => answer = (112 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q262
find (answer : ℕ) s.t.
: (IsGreatest {x | ∃ a m c : ℕ, a + m + c = 12 ∧ x = a*m*c + a*m + m*c + a*c} answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q262
find_all (answer : ℕ) iff
: (IsGreatest {x | ∃ a m c : ℕ, a + m + c = 12 ∧ x = a*m*c + a*m + m*c + a*c} answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
