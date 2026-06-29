import FormalProblemSolving.Basic

/-
## Problem
Evaluate in closed form \[ \sum_{k=1}^n {n \choose k} k^2. \]

## Answer
$n(n+1)2^{n-2}$
-/

-- # Formal Answer
noncomputable abbrev Q2.gt := fun (n : ℕ) (answer : ℕ → ℕ) => answer = fun (n : ℕ) => ((n * (n + (1 : ℕ) : ℕ) : ℕ) * ((2 : ℕ) ^ (n - (2 : ℕ) : ℕ) : ℕ) : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q2
  (n : ℕ)
find (answer : ℕ → ℕ) s.t.
  (hn : n ≥ 2)
: (answer n = ∑ k in Finset.Icc 1 n, Nat.choose n k * k^2)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q2
  (n : ℕ)
find_all (answer : ℕ → ℕ) iff
  (hn : n ≥ 2)
: (answer n = ∑ k in Finset.Icc 1 n, Nat.choose n k * k^2)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
