import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Define a function on the positive integers recursively by $f(1) = 2$, $f(n) = f(n-1) + 1$ if $n$ is even, and $f(n) = f(n-2) + 2$ if $n$ is odd and greater than $1$. What is $f(2017)$?

$ \textbf{(A)}\ 2017 \qquad\textbf{(B)}\ 2018 \qquad\textbf{(C)}\ 4034 \qquad\textbf{(D)}\ 4035 \qquad\textbf{(E)}\ 4036 $

## Answer
\textbf{(B)} 2018
-/

-- # Formal Answer
noncomputable abbrev Q68.gt := fun (answer : ℝ) => answer = (2018 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q68
find (answer : ℝ) s.t.
  (f : ℕ → ℝ)
  (h₀ : f 1 = 2)
  (h₁ : ∀ n, 1 < n ∧ Even n → f n = f (n - 1) + 1)
  (h₂ : ∀ n, 1 < n ∧ Odd n → f n = f (n - 2) + 2)
: (answer = f 2017)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q68
find_all (answer : ℝ) iff
  (f : ℕ → ℝ)
  (h₀ : f 1 = 2)
  (h₁ : ∀ n, 1 < n ∧ Even n → f n = f (n - 1) + 1)
  (h₂ : ∀ n, 1 < n ∧ Odd n → f n = f (n - 2) + 2)
: (answer = f 2017)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
