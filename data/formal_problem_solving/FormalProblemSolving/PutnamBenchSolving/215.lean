import FormalProblemSolving.Basic
open Set Nat Function

/-
## Problem
Let $n$ be a positive integer. Find the number of pairs $P, Q$ of polynomials with real coefficients such that
\[
(P(X))^2 + (Q(X))^2 = X^{2n} + 1
\]
and $\deg P > \deg Q$.

## Answer
fun n ↦ 2 ^ (n + 1)
-/

-- # Formal Answer
noncomputable abbrev Q215.gt := fun (n : ℕ) (answer : ℕ → ℕ) => answer = fun (n : ℕ) => ((2 : ℕ) ^ (n + (1 : ℕ) : ℕ) : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q215
  (n : ℕ)
find (answer : ℕ → ℕ) s.t.
  (npos : n > 0)
: ({a : (Polynomial ℝ) × (Polynomial ℝ) | a.1 ^ 2 + a.2 ^ 2 = Polynomial.X ^ (2 * n) + 1 ∧ a.1.degree > a.2.degree}.ncard = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q215
  (n : ℕ)
find_all (answer : ℕ → ℕ) iff
  (npos : n > 0)
: ({a : (Polynomial ℝ) × (Polynomial ℝ) | a.1 ^ 2 + a.2 ^ 2 = Polynomial.X ^ (2 * n) + 1 ∧ a.1.degree > a.2.degree}.ncard = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
