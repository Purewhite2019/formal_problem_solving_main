import FormalProblemSolving.Basic
open Filter Topology Set

/-
## Problem
There are $2010$ boxes labeled $B_1, B_2, \dots, B_{2010}$, and $2010n$ balls have been distributed among them, for some positive integer $n$. You may redistribute the balls by a sequence of moves, each of which consists of choosing an $i$ and moving \emph{exactly} $i$ balls from box $B_i$ into any one other box. For which values of $n$ is it possible to reach the distribution with exactly $n$ balls in each box, regardless of the initial distribution of balls?

## Answer
{1005}
-/

-- # Formal Answer
noncomputable abbrev Q233.gt := fun (answer : Set ℕ) => answer = ({(1005 : ℕ)} : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q233
find (answer : Set ℕ) s.t.
  (n : ℕ)
  (hn : n > 0)
  (trans : (ℕ → Fin 2010 → ℕ) → ℕ → Prop)
  (htrans : ∀ P T, trans P T ↔ ∀ t : ℕ, t < T → ∃ i j,
            i ≠ j ∧
            P t i ≥ i.1 + 1 ∧ P (t + 1) i = P t i - (i.1 + 1) ∧ P (t + 1) j = P t j + (i.1 + 1) ∧
            ∀ k : Fin 2010, k ≠ i → k ≠ j → P (t + 1) k = P t k)
: ((∀ B, ∑ i, B i = 2010 * n → ∃ᵉ (P) (T), P 0 = B ∧ trans P T ∧ ∀ i, P T i = n) ↔ n ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q233
find_all (answer : Set ℕ) iff
  (n : ℕ)
  (hn : n > 0)
  (trans : (ℕ → Fin 2010 → ℕ) → ℕ → Prop)
  (htrans : ∀ P T, trans P T ↔ ∀ t : ℕ, t < T → ∃ i j,
            i ≠ j ∧
            P t i ≥ i.1 + 1 ∧ P (t + 1) i = P t i - (i.1 + 1) ∧ P (t + 1) j = P t j + (i.1 + 1) ∧
            ∀ k : Fin 2010, k ≠ i → k ≠ j → P (t + 1) k = P t k)
: ((∀ B, ∑ i, B i = 2010 * n → ∃ᵉ (P) (T), P 0 = B ∧ trans P T ∧ ∀ i, P T i = n) ↔ n ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
