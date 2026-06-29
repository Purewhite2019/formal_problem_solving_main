import FormalProblemSolving.Basic
open Topology Filter Polynomial Set

/-
## Problem
For each integer $m$, consider the polynomial
\[P_m(x)=x^4-(2m+4)x^2+(m-2)^2.\] For what values of $m$ is $P_m(x)$
the product of two non-constant polynomials with integer coefficients?

## Answer
{m : ℤ | ∃ k : ℤ, k^2 = m ∨ 2*k^2 = m}
-/

-- # Formal Answer
noncomputable abbrev Q184.gt := fun (answer : Set ℤ) => answer = ({m : ℤ | ∃ (k : ℤ), (k ^ (2 : ℕ) : ℤ) = m ∨ ((2 : ℤ) * (k ^ (2 : ℕ) : ℤ) : ℤ) = m} : Set ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q184
find (answer : Set ℤ) s.t.
  (P : ℤ → Polynomial ℤ)
  (hP : P = fun m : ℤ => (Polynomial.X)^4 - (Polynomial.C (2*m + 4))*(Polynomial.X)^2 + Polynomial.C ((m - 2)^2))
: ({m : ℤ | ∃ a b, P m = a * b ∧
(∃ n ∈ Ici 1, a.coeff n ≠ 0) ∧ (∃ n ∈ Ici 1, b.coeff n ≠ 0)} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q184
find_all (answer : Set ℤ) iff
  (P : ℤ → Polynomial ℤ)
  (hP : P = fun m : ℤ => (Polynomial.X)^4 - (Polynomial.C (2*m + 4))*(Polynomial.X)^2 + Polynomial.C ((m - 2)^2))
: ({m : ℤ | ∃ a b, P m = a * b ∧
(∃ n ∈ Ici 1, a.coeff n ≠ 0) ∧ (∃ n ∈ Ici 1, b.coeff n ≠ 0)} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
