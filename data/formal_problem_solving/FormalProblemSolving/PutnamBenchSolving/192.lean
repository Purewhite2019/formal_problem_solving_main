import FormalProblemSolving.Basic
open MvPolynomial Set

/-
## Problem
For a set $S$ of nonnegative integers, let $r_S(n)$ denote the number of ordered pairs $(s_1,s_2)$ such that $s_1 \in S$, $s_2 \in S$, $s_1 \ne s_2$, and $s_1+s_2=n$. Is it possible to partition the nonnegative integers into two sets $A$ and $B$ in such a way that $r_A(n)=r_B(n)$ for all $n$?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q192.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q192
find (answer : Prop) s.t.
  (r : Set ℕ → ℕ → ℕ)
  (hr : ∀ S n, r S n = ∑' s1 : S, ∑' s2 : S, if (s1 ≠ s2 ∧ s1 + s2 = n) then 1 else 0)
: ((∃ A B : Set ℕ, A ∪ B = ℕ ∧ A ∩ B = ∅ ∧ (∀ n : ℕ, r A n = r B n)) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q192
find_all (answer : Prop) iff
  (r : Set ℕ → ℕ → ℕ)
  (hr : ∀ S n, r S n = ∑' s1 : S, ∑' s2 : S, if (s1 ≠ s2 ∧ s1 + s2 = n) then 1 else 0)
: ((∃ A B : Set ℕ, A ∪ B = ℕ ∧ A ∩ B = ∅ ∧ (∀ n : ℕ, r A n = r B n)) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
