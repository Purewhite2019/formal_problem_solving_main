import FormalProblemSolving.Basic
open Topology MvPolynomial Filter Set

/-
## Problem
Call a subset $S$ of $\{1, 2, \dots, n\}$ \emph{mediocre} if it has the following property: Whenever $a$ and $b$ are elements of $S$ whose average is an integer, that average is also an element of $S$. Let $A(n)$ be the number of mediocre subsets of $\{1,2,\dots,n\}$. [For instance, every subset of $\{1,2,3\}$ except $\{1,3\}$ is mediocre, so $A(3) = 7$.] Find all positive integers $n$ such that $A(n+2) - 2A(n+1) + A(n) = 1$.

## Answer
{n : ℤ | ∃ k ≥ 1, n = 2 ^ k - 1}
-/

-- # Formal Answer
noncomputable abbrev Q227.gt := fun (answer : Set ℤ) => answer = ({n : ℤ | ∃ k ≥ (1 : ℕ), n = (((2 : ℤ) ^ k : ℤ) - (1 : ℤ) : ℤ)} : Set ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q227
find (answer : Set ℤ) s.t.
  (mediocre : ℤ → Set ℤ → Prop)
  (hmediocre : ∀ n S, mediocre n S ↔ (S ⊆ Icc 1 n) ∧ ∀ a ∈ S, ∀ b ∈ S, 2 ∣ a + b → (a + b) / 2 ∈ S)
  (A : ℤ → ℤ)
  (hA : A = fun n ↦ ({S : Set ℤ | mediocre n S}.ncard : ℤ))
: ({n : ℤ | n > 0 ∧ A (n + 2) - 2 * A (n + 1) + A n = 1} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q227
find_all (answer : Set ℤ) iff
  (mediocre : ℤ → Set ℤ → Prop)
  (hmediocre : ∀ n S, mediocre n S ↔ (S ⊆ Icc 1 n) ∧ ∀ a ∈ S, ∀ b ∈ S, 2 ∣ a + b → (a + b) / 2 ∈ S)
  (A : ℤ → ℤ)
  (hA : A = fun n ↦ ({S : Set ℤ | mediocre n S}.ncard : ℤ))
: ({n : ℤ | n > 0 ∧ A (n + 2) - 2 * A (n + 1) + A n = 1} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
