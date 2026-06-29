import FormalProblemSolving.Basic
open Function

/-
## Problem
Define a \emph{selfish} set to be a set which has its own cardinality (number of elements) as an element. Find, with proof, the number of subsets of $\{1,2,\ldots,n\}$ which are \emph{minimal} selfish sets, that is, selfish sets none of whose proper subsets is selfish.

## Answer
the nth Fibonacci number
-/

-- # Formal Answer
noncomputable abbrev Q165.gt := fun (answer : ℕ → ℕ) => answer = (Nat.fib : ℕ → ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q165
find (answer : ℕ → ℕ) s.t.
  (selfish : Finset ℕ → Prop)
  (n : ℕ)
  (hselfish : ∀ s : Finset ℕ, selfish s = (s.card ∈ s))
  (npos : n ≥ 1)
: ({s : Finset ℕ | (s : Set ℕ) ⊆ Set.Icc 1 n ∧ selfish s ∧ (∀ ss : Finset ℕ, ss ⊂ s → ¬selfish ss)}.encard = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q165
find_all (answer : ℕ → ℕ) iff
  (selfish : Finset ℕ → Prop)
  (n : ℕ)
  (hselfish : ∀ s : Finset ℕ, selfish s = (s.card ∈ s))
  (npos : n ≥ 1)
: ({s : Finset ℕ | (s : Set ℕ) ⊆ Set.Icc 1 n ∧ selfish s ∧ (∀ ss : Finset ℕ, ss ⊂ s → ¬selfish ss)}.encard = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
