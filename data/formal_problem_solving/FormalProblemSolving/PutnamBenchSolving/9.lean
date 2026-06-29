import FormalProblemSolving.Basic
open EuclideanGeometry Topology Filter Complex

/-
## Problem
How many orderings of the integers from $1$ to $n$ satisfy the condition that, for every integer $i$ except the first, there exists some earlier integer in the ordering which differs from $i$ by $1$?

## Answer
$2^{n-1}$
-/

-- # Formal Answer
noncomputable abbrev Q9.gt := fun (n : ℕ) (answer : ℕ → ℕ) => answer = fun (n : ℕ) => ((2 : ℕ) ^ (n - (1 : ℕ) : ℕ) : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q9
  (n : ℕ)
find (answer : ℕ → ℕ) s.t.
  (npos : n > 0)
: ({p ∈ permsOfFinset (Finset.Icc 1 n) | ∀ m ∈ Finset.Icc 2 n, ∃ k ∈ Finset.Ico 1 m, p m = p k + 1 ∨ p m = p k - 1}.card = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q9
  (n : ℕ)
find_all (answer : ℕ → ℕ) iff
  (npos : n > 0)
: ({p ∈ permsOfFinset (Finset.Icc 1 n) | ∀ m ∈ Finset.Icc 2 n, ∃ k ∈ Finset.Ico 1 m, p m = p k + 1 ∨ p m = p k - 1}.card = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
