import FormalProblemSolving.Basic
open Nat Filter Topology Real

/-
## Problem
Let $f(n)$ be the number of ways of representing $n$ as a sum of powers of $2$ with no power being used more than $3$ times. For example, $f(7) = 4$ (the representations are $4 + 2 + 1$, $4 + 1 + 1 + 1$, $2 + 2 + 2 + 1$, $2 + 2 + 1 + 1 + 1$). Can we find a real polynomial $p(x)$ such that $f(n) = [p(n)]$, where $[u]$ denotes the greatest integer less than or equal to $u$?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q83.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q83
find (answer : Prop) s.t.
  (f : ℕ+ → ℕ)
  (hf : f = fun (n : ℕ+) ↦
      Set.ncard {M : Multiset ℕ |
        (∀ m ∈ M, ∃ k : ℕ, m = (2 ^ k : ℤ)) ∧
        (∀ m ∈ M, M.count m ≤ 3) ∧
        (M.sum : ℤ) = n})
: (answer ↔ (∃ p : Polynomial ℝ, ∀ n : ℕ+, ⌊p.eval (n : ℝ)⌋ = f n))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q83
find_all (answer : Prop) iff
  (f : ℕ+ → ℕ)
  (hf : f = fun (n : ℕ+) ↦
      Set.ncard {M : Multiset ℕ |
        (∀ m ∈ M, ∃ k : ℕ, m = (2 ^ k : ℤ)) ∧
        (∀ m ∈ M, M.count m ≤ 3) ∧
        (M.sum : ℤ) = n})
: (answer ↔ (∃ p : Polynomial ℝ, ∀ n : ℕ+, ⌊p.eval (n : ℝ)⌋ = f n))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
