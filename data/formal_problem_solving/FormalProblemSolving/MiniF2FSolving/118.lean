import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For what value of $n$ is $i + 2i^2 + 3i^3 + \cdots + ni^n = 48 + 49i$?

Note: here $i = \sqrt { - 1}$.

$\textbf{(A)}\ 24 \qquad \textbf{(B)}\ 48 \qquad \textbf{(C)}\ 49 \qquad \textbf{(D)}\ 97 \qquad \textbf{(E)}\ 98$

## Answer
97
-/

-- # Formal Answer
noncomputable abbrev Q118.gt := fun (answer : ℕ) => answer = (97 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q118
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : 0 < n)
  (h₁ : ∑ k in Finset.Icc 1 n, (↑k * (Complex.I^k)) = 48 + 49 * Complex.I)
: (answer = n)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q118
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : 0 < n)
  (h₁ : ∑ k in Finset.Icc 1 n, (↑k * (Complex.I^k)) = 48 + 49 * Complex.I)
: (answer = n)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
