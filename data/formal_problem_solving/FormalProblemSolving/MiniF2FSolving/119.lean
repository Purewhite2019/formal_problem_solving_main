import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sequence $S_1, S_2, S_3, \cdots, S_{10}$ has the property that every term beginning with the third is the sum of the previous two.  That is, $ S_n = S_{n-2} + S_{n-1} \text{ for } n \ge 3. $ Suppose that $S_9 = 110$ and $S_7 = 42$.  What is $S_4$?

$ \textbf{(A)}\ 4\qquad\textbf{(B)}\ 6\qquad\textbf{(C)}\ 10\qquad\textbf{(D)}\ 12\qquad\textbf{(E)}\ 16\qquad $

## Answer
10
-/

-- # Formal Answer
noncomputable abbrev Q119.gt := fun (answer : ℝ) => answer = (10 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q119
find (answer : ℝ) s.t.
  (s : ℕ → ℝ)
  (h₀ : ∀ n, 3 ≤ n → n ≤ 10 → s n = s (n - 1) + s (n - 2))
  (h₁ : s 9 = 110)
  (h₂ : s 7 = 42)
: (answer = s 4)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q119
find_all (answer : ℝ) iff
  (s : ℕ → ℝ)
  (h₀ : ∀ n, 3 ≤ n → n ≤ 10 → s n = s (n - 1) + s (n - 2))
  (h₁ : s 9 = 110)
  (h₂ : s 7 = 42)
: (answer = s 4)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
