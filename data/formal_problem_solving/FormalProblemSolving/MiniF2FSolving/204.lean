import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
There is a unique positive integer $n$ such that$\log_2{(\log_{16}{n})} = \log_4{(\log_4{n})}.$What is the sum of the digits of $n?$

$\textbf{(A) } 4 \qquad \textbf{(B) } 7 \qquad \textbf{(C) } 8 \qquad \textbf{(D) } 11 \qquad \textbf{(E) } 13$

## Answer
13
-/

-- # Formal Answer
noncomputable abbrev Q204.gt := fun (answer : ℕ) => answer = (13 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q204
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : 1 < n)
  (h₁ : Real.logb 2 (Real.logb 16 n) = Real.logb 4 (Real.logb 4 n))
  (h_unique : ∀ (m : ℕ), 1 < m → Real.logb 2 (Real.logb 16 m) = Real.logb 4 (Real.logb 4 m) → m = n)
: (answer = (Nat.digits 10 n).sum)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q204
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : 1 < n)
  (h₁ : Real.logb 2 (Real.logb 16 n) = Real.logb 4 (Real.logb 4 n))
  (h_unique : ∀ (m : ℕ), 1 < m → Real.logb 2 (Real.logb 16 m) = Real.logb 4 (Real.logb 4 m) → m = n)
: (answer = (Nat.digits 10 n).sum)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
