import FormalProblemSolving.Basic

/-
## Problem
Let $A_1=0$ and $A_2=1$. For $n>2$, the number $A_n$ is defined by concatenating the decimal expansions of $A_{n-1}$ and $A_{n-2}$ from left to right.  For example $A_3=A_2 A_1=10$, $A_4=A_3 A_2 = 101$, $A_5=A_4 A_3 = 10110$, and so forth.  Determine all $n$ such that $11$ divides $A_n$.

## Answer
all $n$ such that $n \equiv 1 \pmod{6}$
-/

-- # Formal Answer
noncomputable abbrev Q174.gt := fun (answer : Set ℕ) => answer = ({n : ℕ | n ≡ (1 : ℕ) [MOD (6 : ℕ)]} : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q174
find (answer : Set ℕ) s.t.
  (A : ℕ → List ℕ)
  (hA1 : A 1 = [0])
  (hA2 : A 2 = [1])
  (hA : ∀ n > 0, A (n + 2) = A (n + 1) ++ A n)
: ({n | 1 ≤ n ∧ 11 ∣ Nat.ofDigits 10 (A n).reverse} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q174
find_all (answer : Set ℕ) iff
  (A : ℕ → List ℕ)
  (hA1 : A 1 = [0])
  (hA2 : A 2 = [1])
  (hA : ∀ n > 0, A (n + 2) = A (n + 1) ++ A n)
: ({n | 1 ≤ n ∧ 11 ∣ Nat.ofDigits 10 (A n).reverse} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
