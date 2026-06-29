import FormalProblemSolving.Basic
open Set Polynomial

/-
## Problem
Let $p(x) = 2(x^6 + 1) + 4(x^5 + x) + 3(x^4 + x^2) + 5x^3$. For $k$ with $0 < k < 5$, let
\[
I_k = \int_0^{\infty} \frac{x^k}{p(x)} \, dx.
\]
For which $k$ is $I_k$ smallest?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q52.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q52
find (answer : ℕ) s.t.
  (p : Polynomial ℝ)
  (hp : p = 2 * (Polynomial.X ^ 6 + 1) + 4 * (Polynomial.X ^ 5 + Polynomial.X) + 3 * (Polynomial.X ^ 4 + Polynomial.X ^ 2) + 5 * Polynomial.X ^ 3)
  (I : ℕ → ℝ)
  (hI : I = fun k ↦ ∫ x in Ioi 0, x ^ k / p.eval x)
: (IsLeast {y | ∃ k ∈ Ioo 0 5, I k = y} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q52
find_all (answer : ℕ) iff
  (p : Polynomial ℝ)
  (hp : p = 2 * (Polynomial.X ^ 6 + 1) + 4 * (Polynomial.X ^ 5 + Polynomial.X) + 3 * (Polynomial.X ^ 4 + Polynomial.X ^ 2) + 5 * Polynomial.X ^ 3)
  (I : ℕ → ℝ)
  (hI : I = fun k ↦ ∫ x in Ioi 0, x ^ k / p.eval x)
: (IsLeast {y | ∃ k ∈ Ioo 0 5, I k = y} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
