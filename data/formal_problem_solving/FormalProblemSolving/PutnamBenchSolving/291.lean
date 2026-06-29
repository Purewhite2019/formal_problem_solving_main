import FormalProblemSolving.Basic

/-
## Problem
Let $k$ be a nonnegative integer. Evaluate
\[
\sum_{j=0}^k 2^{k-j} \binom{k+j}{j}.
\]

## Answer
4^k
-/

-- # Formal Answer
noncomputable abbrev Q291.gt := fun (k : ℕ) (answer : ℕ → ℕ) => answer = fun (k : ℕ) => ((4 : ℕ) ^ k : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q291
  (k : ℕ)
find (answer : ℕ → ℕ) s.t.
: ((∑ j in Finset.Icc 0 k, 2 ^ (k - j) * Nat.choose (k + j) j) = answer k)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q291
  (k : ℕ)
find_all (answer : ℕ → ℕ) iff
: ((∑ j in Finset.Icc 0 k, 2 ^ (k - j) * Nat.choose (k + j) j) = answer k)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
