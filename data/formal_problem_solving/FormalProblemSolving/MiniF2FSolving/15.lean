import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Which of the following is equal to the [[product]]
$\frac{8}{4}\cdot\frac{12}{8}\cdot\frac{16}{12}\cdot\cdots\cdot\frac{4n+4}{4n}\cdot\cdots\cdot\frac{2008}{2004}?$

$\textbf{(A)}\ 251\qquad\textbf{(B)}\ 502\qquad\textbf{(C)}\ 1004\qquad\textbf{(D)}\ 2008\qquad\textbf{(E)}\ 4016$

## Answer
502
-/

-- # Formal Answer
noncomputable abbrev Q15.gt := fun (answer : ℝ) => answer = (502 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q15
find (answer : ℝ) s.t.
  (S : Finset ℕ)
  (hS : S = Finset.Icc 1 501)
: (∏ k in S, ((4 : ℝ) * k + 4) / (4 * k) = answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q15
find_all (answer : ℝ) iff
  (S : Finset ℕ)
  (hS : S = Finset.Icc 1 501)
: (∏ k in S, ((4 : ℝ) * k + 4) / (4 * k) = answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
