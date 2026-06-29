import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The first two terms of a sequence are $a_1 = 1$ and $a_2 = \frac {1}{\sqrt3}$. For $n\ge1$,

<center>$a_{n + 2} = \frac {a_n + a_{n + 1}}{1 - a_na_{n + 1}}.$</center>

What is $|a_{2009}|$?

$\textbf{(A)}\ 0\qquad \textbf{(B)}\ 2 - \sqrt3\qquad \textbf{(C)}\ \frac {1}{\sqrt3}\qquad \textbf{(D)}\ 1\qquad \textbf{(E)}\ 2 + \sqrt3$

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q156.gt := fun (answer : ℝ) => answer = (0 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q156
find (answer : ℝ) s.t.
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 1 / Real.sqrt 3)
  (h₂ : ∀ n, 1 ≤ n → a (n + 2) = (a n + a (n + 1)) / (1 - a n * a (n + 1)))
: (answer = abs (a 2009))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q156
find_all (answer : ℝ) iff
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 1 / Real.sqrt 3)
  (h₂ : ∀ n, 1 ≤ n → a (n + 2) = (a n + a (n + 1)) / (1 - a n * a (n + 1)))
: (answer = abs (a 2009))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
