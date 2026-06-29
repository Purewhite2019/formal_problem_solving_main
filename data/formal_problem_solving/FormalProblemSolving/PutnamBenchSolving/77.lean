import FormalProblemSolving.Basic
open Set Function Filter Topology Polynomial Real

/-
## Problem
Let $A(x, y)$ denote the number of points $(m, n)$ with integer coordinates $m$ and $n$ where $m^2 + n^2 \le x^2 + y^2$. Also, let $g = \sum_{k = 0}^{\infty} e^{-k^2}$. Express the value $$\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} A(x, y)e^{-x^2 - y^2} dx dy$$ as a polynomial in $g$.

## Answer
C Real.pi * (2*X - 1)^2
-/

-- # Formal Answer
noncomputable abbrev Q77.gt := fun (answer : Polynomial ℝ) => answer =
  ((((Polynomial.C : ℝ →+* ℝ[X]) : ℝ → ℝ[X]) (π : ℝ) : ℝ[X]) *
      ((((2 : ℝ[X]) * (Polynomial.X : ℝ[X]) : ℝ[X]) - (1 : ℝ[X]) : ℝ[X]) ^ (2 : ℕ) : ℝ[X]) :
    ℝ[X])

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q77
find (answer : Polynomial ℝ) s.t.
  (A : ℝ × ℝ → ℕ)
  (g : ℝ)
  (I : ℝ)
  (hA : A = fun (x, y) => {a : ℤ × ℤ | a.1^2 + a.2^2 ≤ x^2 + y^2}.ncard)
  (hg : g = ∑' k : ℕ, Real.exp (-k^2))
  (hI : I = ∫ y : ℝ, ∫ x : ℝ, A (x, y) * Real.exp (-x^2 - y^2))
: (I = answer.eval g)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q77
find_all (answer : Polynomial ℝ) iff
  (A : ℝ × ℝ → ℕ)
  (g : ℝ)
  (I : ℝ)
  (hA : A = fun (x, y) => {a : ℤ × ℤ | a.1^2 + a.2^2 ≤ x^2 + y^2}.ncard)
  (hg : g = ∑' k : ℕ, Real.exp (-k^2))
  (hI : I = ∫ y : ℝ, ∫ x : ℝ, A (x, y) * Real.exp (-x^2 - y^2))
: (I = answer.eval g)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
