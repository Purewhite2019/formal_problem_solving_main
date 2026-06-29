import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sum of the two 5-digit numbers $AMC10$ and $AMC12$ is $123422$. What is $A+M+C$?

$ \mathrm{(A) \ } 10\qquad \mathrm{(B) \ } 11\qquad \mathrm{(C) \ } 12\qquad \mathrm{(D) \ } 13\qquad \mathrm{(E) \ } 14 $

## Answer
14
-/

-- # Formal Answer
noncomputable abbrev Q248.gt := fun (answer : ℕ) => answer = (14 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q248
find (answer : ℕ) s.t.
  (A : ℕ)
  (M : ℕ)
  (C : ℕ)
  (h₀ : A ≤ 9 ∧ M ≤ 9 ∧ C ≤ 9)
  (h₁ : Nat.ofDigits 10 [0,1,C,M,A] + Nat.ofDigits 10 [2,1,C,M,A] = 123422)
: (answer = A + M + C)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q248
find_all (answer : ℕ) iff
  (A : ℕ)
  (M : ℕ)
  (C : ℕ)
  (h₀ : A ≤ 9 ∧ M ≤ 9 ∧ C ≤ 9)
  (h₁ : Nat.ofDigits 10 [0,1,C,M,A] + Nat.ofDigits 10 [2,1,C,M,A] = 123422)
: (answer = A + M + C)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
