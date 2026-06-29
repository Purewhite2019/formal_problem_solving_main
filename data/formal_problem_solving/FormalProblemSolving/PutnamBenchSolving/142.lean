import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
For each integer $n \geq 0$, let $S(n)=n-m^2$, where $m$ is the greatest integer with $m^2 \leq n$. Define a sequence $(a_k)_{k=0}^\infty$ by $a_0=A$ and $a_{k+1}=a_k+S(a_k)$ for $k \geq 0$. For what positive integers $A$ is this sequence eventually constant?

## Answer
the set of positive integers $A$ such that $A$ is a perfect square
-/

-- # Formal Answer
noncomputable abbrev Q142.gt := fun (A : ℤ) (answer : Set ℤ) => answer = ({A : ℤ | ∃ x > (0 : ℤ), A = (x ^ (2 : ℕ) : ℤ)} : Set ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q142
  (A : ℤ)
find (answer : Set ℤ) s.t.
  (m : ℤ → ℤ)
  (S : ℤ → ℤ)
  (a : ℕ → ℤ)
  (hm : ∀ n, 0 ≤ n → (m n) ^ 2 ≤ n ∧ (∀ m' : ℤ, m' ^ 2 ≤ n → m' ≤ m n))
  (hS : ∀ n, 0 ≤ n → S n = n - (m n) ^ 2)
  (ha : a 0 = A ∧ (∀ k, a (k + 1) = a k + S (a k)))
  (hA : A > 0)
: ((∃ (K : ℕ) (c : ℤ), ∀ k ≥ K, a k = c) ↔ A ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q142
  (A : ℤ)
find_all (answer : Set ℤ) iff
  (m : ℤ → ℤ)
  (S : ℤ → ℤ)
  (a : ℕ → ℤ)
  (hm : ∀ n, 0 ≤ n → (m n) ^ 2 ≤ n ∧ (∀ m' : ℤ, m' ^ 2 ≤ n → m' ≤ m n))
  (hS : ∀ n, 0 ≤ n → S n = n - (m n) ^ 2)
  (ha : a 0 = A ∧ (∀ k, a (k + 1) = a k + S (a k)))
  (hA : A > 0)
: ((∃ (K : ℕ) (c : ℤ), ∀ k ≥ K, a k = c) ↔ A ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
