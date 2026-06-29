import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The first four terms of an arithmetic sequence are $p$, $9$, $3p-q$, and $3p+q$. What is the $2010^\text{th}$ term of this sequence?

$\textbf{(A)}\ 8041 \qquad \textbf{(B)}\ 8043 \qquad \textbf{(C)}\ 8045 \qquad \textbf{(D)}\ 8047 \qquad \textbf{(E)}\ 8049$

## Answer
8041
-/

-- # Formal Answer
noncomputable abbrev Q147.gt := fun (answer : ℝ) => answer = (8041 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q147
find (answer : ℝ) s.t.
  (p : ℝ)
  (q : ℝ)
  (a : ℕ → ℝ)
  (h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)
  (h₁ : a 1 = p)
  (h₂ : a 2 = 9)
  (h₃ : a 3 = 3 * p - q)
  (h₄ : a 4 = 3 * p + q)
: (answer = a 2010)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q147
find_all (answer : ℝ) iff
  (p : ℝ)
  (q : ℝ)
  (a : ℕ → ℝ)
  (h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)
  (h₁ : a 1 = p)
  (h₂ : a 2 = 9)
  (h₃ : a 3 = 3 * p - q)
  (h₄ : a 4 = 3 * p + q)
: (answer = a 2010)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
