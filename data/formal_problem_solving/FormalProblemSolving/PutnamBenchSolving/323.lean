import FormalProblemSolving.Basic
open Nat Topology Filter

/-
## Problem
Determine which positive integers $n$ have the following property: For all integers $m$ that are relatively prime to $n$, there exists a permutation $\pi:\{1,2,\dots,n\} \to \{1,2,\dots,n\}$ such that $\pi(\pi(k)) \equiv mk \pmod{n}$ for all $k \in \{1,2,\dots,n\}$.

## Answer
{n : ℕ | n = 1 ∨ n ≡ 2 [MOD 4]}
-/

-- # Formal Answer
noncomputable abbrev Q323.gt := fun (answer : Set ℕ) => answer = ({n : ℕ | n = (1 : ℕ) ∨ n ≡ (2 : ℕ) [MOD (4 : ℕ)]} : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q323
find (answer : Set ℕ) s.t.
: answer = {n : ℕ | 0 < n ∧ (∀ m : ℤ, IsRelPrime m n → ∃ p : Equiv.Perm (Fin n),
    ∀ k : Fin n, (p (p k)).1 + 1 ≡ m * (k.1 + 1) [ZMOD n])}
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q323
find_all (answer : Set ℕ) iff
: answer = {n : ℕ | 0 < n ∧ (∀ m : ℤ, IsRelPrime m n → ∃ p : Equiv.Perm (Fin n),
    ∀ k : Fin n, (p (p k)).1 + 1 ≡ m * (k.1 + 1) [ZMOD n])}
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
