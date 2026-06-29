import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $n = 3^{17} + 3^{10}$. It is known that $11$ divides into $n+1$. If $n$ can be written in base $10$ as $ABCACCBAB$, where $A,B,C$ are distinct digits such that $A$ and $C$ are odd and $B$ is not divisible by $3$, find $100A + 10B + C$.

## Answer
129
-/

-- # Formal Answer
noncomputable abbrev Q263.gt := fun (answer : ℕ) => answer = (129 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q263
find (answer : ℕ) s.t.
  (n : ℕ)
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (h₀ : n = 3^17 + 3^10)
  (h₁ : 11 ∣ (n + 1))
  (h₂ : A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (h₃ : A < 10 ∧ B < 10 ∧ C < 10)
  (h₄ : Odd A ∧ Odd C)
  (h₅ : ¬ 3 ∣ B)
  (h₆ : Nat.digits 10 n = [B,A,B,C,C,A,C,B,A])
: (answer = 100 * A + 10 * B + C)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q263
find_all (answer : ℕ) iff
  (n : ℕ)
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (h₀ : n = 3^17 + 3^10)
  (h₁ : 11 ∣ (n + 1))
  (h₂ : A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (h₃ : A < 10 ∧ B < 10 ∧ C < 10)
  (h₄ : Odd A ∧ Odd C)
  (h₅ : ¬ 3 ∣ B)
  (h₆ : Nat.digits 10 n = [B,A,B,C,C,A,C,B,A])
: (answer = 100 * A + 10 * B + C)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
