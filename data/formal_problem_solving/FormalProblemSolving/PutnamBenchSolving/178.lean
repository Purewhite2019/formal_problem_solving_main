import FormalProblemSolving.Basic
open Set Function Metric

/-
## Problem
Let $N$ be the positive integer with 1998 decimal digits, all of them 1; that is, \[N=1111\cdots 11.\] Find the thousandth digit after the decimal point of $\sqrt N$.

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q178.gt := fun (answer : ℕ) => answer = (1 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q178
find (answer : ℕ) s.t.
  (N : ℕ)
  (hN : N = ∑ i in Finset.range 1998, 10^i)
: (answer = (Nat.floor (10^1000 * Real.sqrt N)) % 10)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q178
find_all (answer : ℕ) iff
  (N : ℕ)
  (hN : N = ∑ i in Finset.range 1998, 10^i)
: (answer = (Nat.floor (10^1000 * Real.sqrt N)) % 10)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
