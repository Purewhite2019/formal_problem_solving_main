import FormalProblemSolving.Basic
open Topology Filter Nat

/-
## Problem
A \emph{base $10$ over-expansion} of a positive integer $N$ is an expression of the form
\[
N = d_k 10^k + d_{k-1} 10^{k-1} + \cdots + d_0 10^0
\]
with $d_k \neq 0$ and $d_i \in \{0,1,2,\dots,10\}$ for all $i$. For instance, the integer $N = 10$ has two base $10$ over-expansions: $10 = 10 \cdot 10^0$ and the usual base $10$ expansion $10 = 1 \cdot 10^1 + 0 \cdot 10^0$. Which positive integers have a unique base $10$ over-expansion?

## Answer
the set of positive integers that do not contain the digit 0 in their base 10 representation
-/

-- # Formal Answer
noncomputable abbrev Q253.gt := fun (answer : Set ℕ) => answer = ({n : ℕ | n > (0 : ℕ) ∧ ¬∃ a ∈ (Nat.digits (10 : ℕ) n : List ℕ), a = (0 : ℕ)} : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q253
find (answer : Set ℕ) s.t.
  (overexpansion : ℕ → List ℕ → Prop)
  (hoverexpansion : overexpansion = fun N d ↦ N = ∑ i : Fin d.length, (d.get i) * 10 ^ i.1 ∧ d.getLastI ≠ 0 ∧ ∀ a ∈ d, a ∈ Finset.range 11)
  (S : Set ℕ)
  (hS : ∀ N : ℕ, N ∈ S ↔ N > 0 ∧ ∃! d : List ℕ, overexpansion N d)
: (S = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q253
find_all (answer : Set ℕ) iff
  (overexpansion : ℕ → List ℕ → Prop)
  (hoverexpansion : overexpansion = fun N d ↦ N = ∑ i : Fin d.length, (d.get i) * 10 ^ i.1 ∧ d.getLastI ≠ 0 ∧ ∀ a ∈ d, a ∈ Finset.range 11)
  (S : Set ℕ)
  (hS : ∀ N : ℕ, N ∈ S ↔ N > 0 ∧ ∃! d : List ℕ, overexpansion N d)
: (S = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
