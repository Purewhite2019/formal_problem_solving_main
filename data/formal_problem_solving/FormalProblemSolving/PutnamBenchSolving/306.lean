import FormalProblemSolving.Basic
open Polynomial

/-
## Problem
Let $n$ be an integer with $n \geq 2$. Over all real polynomials $p(x)$ of degree $n$, what is the largest possible number of negative coefficients of $p(x)^2$?

## Answer
$2n - 2$
-/

-- # Formal Answer
noncomputable abbrev Q306.gt := fun (n : ℕ) (answer : ℕ → ℕ) => answer = fun (n : ℕ) => (((2 : ℕ) * n : ℕ) - (2 : ℕ) : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q306
  (n : ℕ)
find (answer : ℕ → ℕ) s.t.
  (hn : n ≥ 2)
  (S : Set ℝ[X])
  (hS : S = {P | natDegree P = n})
  (negs : ℝ[X] → ℕ)
  (hnegs : ∀ P : ℝ[X], negs P = ∑ i in Finset.range (P.natDegree + 1), if P.coeff i < 0 then 1 else 0)
: (sSup {negs (P^2) | P ∈ S} = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q306
  (n : ℕ)
find_all (answer : ℕ → ℕ) iff
  (hn : n ≥ 2)
  (S : Set ℝ[X])
  (hS : S = {P | natDegree P = n})
  (negs : ℝ[X] → ℕ)
  (hnegs : ∀ P : ℝ[X], negs P = ∑ i in Finset.range (P.natDegree + 1), if P.coeff i < 0 then 1 else 0)
: (sSup {negs (P^2) | P ∈ S} = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
