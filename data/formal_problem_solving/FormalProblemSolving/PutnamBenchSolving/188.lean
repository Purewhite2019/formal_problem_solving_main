import FormalProblemSolving.Basic
open Nat

/-
## Problem
Let $k$ be a fixed positive integer. The $n$-th derivative of $\frac{1}{x^k-1}$ has the form $\frac{P_n(x)}{(x^k-1)^{n+1}}$ where $P_n(x)$ is a polynomial. Find $P_n(1)$.

## Answer
(-k)^n * n!
-/

-- # Formal Answer
noncomputable abbrev Q188.gt := fun (k : ℕ) (answer : ℕ → ℕ → ℝ) => answer = fun (k n : ℕ) => (((-(↑k : ℝ) : ℝ) ^ n : ℝ) * (↑(n ! : ℕ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q188
  (k : ℕ)
find (answer : ℕ → ℕ → ℝ) s.t.
  (P : ℕ → Polynomial ℝ)
  (kpos : k > 0)
  (Pderiv : ∀ n x, iteratedDeriv n (fun x' : ℝ => 1 / (x' ^ k - 1)) x = ((P n).eval x) / ((x ^ k - 1) ^ (n + 1)))
: (∀ n, (P n).eval 1 = answer k n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q188
  (k : ℕ)
find_all (answer : ℕ → ℕ → ℝ) iff
  (P : ℕ → Polynomial ℝ)
  (kpos : k > 0)
  (Pderiv : ∀ n x, iteratedDeriv n (fun x' : ℝ => 1 / (x' ^ k - 1)) x = ((P n).eval x) / ((x ^ k - 1) ^ (n + 1)))
: (∀ n, (P n).eval 1 = answer k n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
