import FormalProblemSolving.Basic
open intervalIntegral MvPolynomial Real

/-
## Problem
Say that a polynomial with real coefficients in two variables, $x,y$, is \emph{balanced} if the average value of the polynomial on each circle centered at the origin is $0$. The balanced polynomials of degree at most $2009$ form a vector space $V$ over $\mathbb{R}$. Find the dimension of $V$.

## Answer
2020050
-/

-- # Formal Answer
noncomputable abbrev Q228.gt := fun (answer : ℕ) => answer = (2020050 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q228
find (answer : ℕ) s.t.
  (IsBalanced : MvPolynomial (Fin 2) ℝ → Prop)
  (IsBalanced_def : ∀ P, IsBalanced P ↔ ∀ r > 0,
      (∫ t in (0 : ℝ)..(2 * π), eval ![r * cos t, r * sin t] P) / (2 * π * r) = 0)
  (V : Submodule ℝ (MvPolynomial (Fin 2) ℝ))
  (V_def : ∀ P, P ∈ V ↔ IsBalanced P ∧ P.totalDegree ≤ 2009)
: (Module.rank ℝ V = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q228
find_all (answer : ℕ) iff
  (IsBalanced : MvPolynomial (Fin 2) ℝ → Prop)
  (IsBalanced_def : ∀ P, IsBalanced P ↔ ∀ r > 0,
      (∫ t in (0 : ℝ)..(2 * π), eval ![r * cos t, r * sin t] P) / (2 * π * r) = 0)
  (V : Submodule ℝ (MvPolynomial (Fin 2) ℝ))
  (V_def : ∀ P, P ∈ V ↔ IsBalanced P ∧ P.totalDegree ≤ 2009)
: (Module.rank ℝ V = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
