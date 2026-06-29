import FormalProblemSolving.Basic
open Set Filter Topology Real Polynomial

/-
## Problem
If $p(x)= a_0 + a_1 x + \cdots + a_m x^m$ is a polynomial with real coefficients $a_i$, then set
\[
\Gamma(p(x)) = a_0^2 + a_1^2 + \cdots + a_m^2.
\]
Let $F(x) = 3x^2+7x+2$. Find, with proof, a polynomial $g(x)$ with real coefficients such that
\begin{enumerate}
\item[(i)] $g(0)=1$, and
\item[(ii)] $\Gamma(f(x)^n) = \Gamma(g(x)^n)$
\end{enumerate}
for every integer $n \geq 1$.

## Answer
6x^2 + 5x + 1
-/

-- # Formal Answer
noncomputable abbrev Q97.gt := fun (answer : Polynomial ℝ) => answer =
  ((((6 : ℝ[X]) * ((Polynomial.X : ℝ[X]) ^ (2 : ℕ) : ℝ[X]) : ℝ[X]) + ((5 : ℝ[X]) * (Polynomial.X : ℝ[X]) : ℝ[X]) :
        ℝ[X]) +
      (1 : ℝ[X]) :
    ℝ[X])

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q97
find (answer : Polynomial ℝ) s.t.
  (Γ : Polynomial ℝ → ℝ)
  (f : Polynomial ℝ)
  (hΓ : Γ = fun p ↦ ∑ k in Finset.range (p.natDegree + 1), coeff p k ^ 2)
  (hf : f = 3 * Polynomial.X ^ 2 + 7 * Polynomial.X + 2)
: (let g := answer;
  g.eval 0 = 1 ∧ ∀ n : ℕ, n ≥ 1 → Γ (f ^ n) = Γ (g ^ n))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q97
find_all (answer : Polynomial ℝ) iff
  (Γ : Polynomial ℝ → ℝ)
  (f : Polynomial ℝ)
  (hΓ : Γ = fun p ↦ ∑ k in Finset.range (p.natDegree + 1), coeff p k ^ 2)
  (hf : f = 3 * Polynomial.X ^ 2 + 7 * Polynomial.X + 2)
: (let g := answer;
  g.eval 0 = 1 ∧ ∀ n : ℕ, n ≥ 1 → Γ (f ^ n) = Γ (g ^ n))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
