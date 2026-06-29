import FormalProblemSolving.Basic
open Polynomial Real Complex

/-
## Problem
Let $n = 2m$, where $m$ is an odd integer greater than 1. Let $\theta = e^{2\pi i/n}$. Expression $(1 - \theta)^{-1}$ explicitly as a polynomial in $\theta$ \[ a_k \theta^k  + a_{k-1}\theta^{k-1} + \dots + a_1\theta + a_0\], with integer coefficients $a_i$.

## Answer
$\sum_{j=0}^{(m-1)/2} \theta^{2j+1}$
-/

-- # Formal Answer
noncomputable abbrev Q36.gt := fun (m : ℕ) (answer : ℕ → Polynomial ℤ) => answer = fun (m : ℕ) =>
  ∑ j ∈ (Finset.range ((m - (1 : ℕ) : ℕ) / (2 : ℕ) : ℕ) : Finset ℕ),
    ((Polynomial.X : ℤ[X]) ^ (((2 : ℕ) * j : ℕ) + (1 : ℕ) : ℕ) : ℤ[X])

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q36
  (m : ℕ)
find (answer : ℕ → Polynomial ℤ) s.t.
  (hm : Odd m ∧ m > 1)
  (θ : ℂ)
  (hθ : θ = cexp (2 * Real.pi * I / (2 * m)))
: (1/(1 - θ) = Polynomial.aeval θ (answer m))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q36
  (m : ℕ)
find_all (answer : ℕ → Polynomial ℤ) iff
  (hm : Odd m ∧ m > 1)
  (θ : ℂ)
  (hθ : θ = cexp (2 * Real.pi * I / (2 * m)))
: (1/(1 - θ) = Polynomial.aeval θ (answer m))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
