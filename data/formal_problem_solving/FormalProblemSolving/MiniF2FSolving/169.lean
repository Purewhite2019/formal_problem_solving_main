import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
There are integers $a, b,$ and $c,$ each greater than $1,$ such that

$\sqrt[a]{N\sqrt[b]{N\sqrt[c]{N}}} = \sqrt[36]{N^{25}}$

for all $N \neq 1$. What is $b$?

$\textbf{(A) } 2 \qquad \textbf{(B) } 3 \qquad \textbf{(C) } 4 \qquad \textbf{(D) } 5 \qquad \textbf{(E) } 6$

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q169.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q169
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (h₁ : 1 < a ∧ 1 < b ∧ 1 < c)
  (h₂ : ∀ (n : NNReal) (h₀ : n ≠ 1),
    ((n : ℝ) * (((n : ℝ) * ((n : ℝ) ^ (1 / (c : ℝ)))) ^ (1 / (b : ℝ)))) ^ (1 / (a : ℝ)) = ((n : ℝ) ^ 25) ^ (1 / 36))
: (answer = b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q169
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (h₁ : 1 < a ∧ 1 < b ∧ 1 < c)
  (h₂ : ∀ (n : NNReal) (h₀ : n ≠ 1),
    ((n : ℝ) * (((n : ℝ) * ((n : ℝ) ^ (1 / (c : ℝ)))) ^ (1 / (b : ℝ)))) ^ (1 / (a : ℝ)) = ((n : ℝ) ^ 25) ^ (1 / 36))
: (answer = b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
