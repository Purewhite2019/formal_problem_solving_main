import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Integers $x$ and $y$ with $x>y>0$ satisfy $x+y+xy=80$. What is $x$?

$ \textbf{(A)}\ 8 \qquad\textbf{(B)}\ 10 \qquad\textbf{(C)}\ 15 \qquad\textbf{(D)}\ 18 \qquad\textbf{(E)}\ 26$

## Answer
26
-/

-- # Formal Answer
noncomputable abbrev Q2.gt := fun (answer : ℤ) => answer = (26 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q2
find (answer : ℤ) s.t.
  (x : ℤ)
  (y : ℤ)
  (h₀ : 0 < y)
  (h₁ : y < x)
  (h₂ : x + y + (x * y) = 80)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q2
find_all (answer : ℤ) iff
  (x : ℤ)
  (y : ℤ)
  (h₀ : 0 < y)
  (h₁ : y < x)
  (h₂ : x + y + (x * y) = 80)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
