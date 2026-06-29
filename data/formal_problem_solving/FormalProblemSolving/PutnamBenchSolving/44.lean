import FormalProblemSolving.Basic
open Polynomial Filter Topology ProbabilityTheory MeasureTheory

/-
## Problem
Find $$\sum_{k=0}^{n} (-1)^k {n \choose k} (x - k)^n.$$

## Answer
fun n => C (Nat.factorial n)
-/

-- # Formal Answer
noncomputable abbrev Q44.gt := fun (answer : ℕ → Polynomial ℤ) => answer = fun (n : ℕ) => (((Polynomial.C : ℤ →+* ℤ[X]) : ℤ → ℤ[X]) (↑(n.factorial : ℕ) : ℤ) : ℤ[X])

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q44
find (answer : ℕ → Polynomial ℤ) s.t.
: (∀ n : ℕ, ∑ k in Finset.range (n + 1), C ((-(1 : ℤ))^k * Nat.choose n k) * (X - (C (k : ℤ)))^n = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q44
find_all (answer : ℕ → Polynomial ℤ) iff
: (∀ n : ℕ, ∑ k in Finset.range (n + 1), C ((-(1 : ℤ))^k * Nat.choose n k) * (X - (C (k : ℤ)))^n = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
