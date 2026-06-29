import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $|x - 2| = p$, where $x < 2$, then $x - p =$

$ \textbf{(A)} \ -2 \qquad \textbf{(B)} \ 2 \qquad \textbf{(C)} \ 2-2p \qquad \textbf{(D)} \ 2p-2 \qquad \textbf{(E)} \ |2p-2|  $

## Answer
(C) 2-2p
-/

-- # Formal Answer
noncomputable abbrev Q37.gt := fun (p : ℝ) (answer : ℝ) => answer = ((2 : ℝ) - ((2 : ℝ) * p : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q37
  (p : ℝ)
find (answer : ℝ) s.t.
  (x : ℝ)
  (h₀ : x < 2)
  (h₁ : abs (x - 2) = p)
: (answer = x - p)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q37
  (p : ℝ)
find_all (answer : ℝ) iff
  (x : ℝ)
  (h₀ : x < 2)
  (h₁ : abs (x - 2) = p)
: (answer = x - p)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
