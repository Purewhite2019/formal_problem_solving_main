import FormalProblemSolving.Basic
open MvPolynomial Real

/-
## Problem
Let $P$ be a polynomial, with real coefficients, in three variables and $F$ be a function of two variables such that
\[
P(ux, uy, uz) = u^2 F(y-x,z-x) \quad \mbox{for all real $x,y,z,u$},
\]
and such that $P(1,0,0)=4$, $P(0,1,0)=5$, and $P(0,0,1)=6$. Also let $A,B,C$ be complex numbers with $P(A,B,C)=0$ and $|B-A|=10$. Find $|C-A|$.

## Answer
$\frac{5}{3}\sqrt{30}$
-/

-- # Formal Answer
noncomputable abbrev Q111.gt := fun (answer : ℂ) => answer = ((5 / 3 : ℂ) * (↑(√(30 : ℝ) : ℝ) : ℂ) : ℂ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q111
find (answer : ℂ) s.t.
  (P : MvPolynomial (Fin 3) ℂ)
  (hPreal : ∀ i : Fin 3 →₀ ℕ, (coeff i P).im = 0)
  (F : ℝ → ℝ → ℝ)
  (vars : ℂ → ℂ → ℂ → (Fin 3 → ℂ))
  (hvars : vars = fun a b c ↦ fun i ↦ ite (i = 0) a (ite (i = 1) b c))
  (h : ∀ x y z u : ℝ, eval (vars (u * x) (u * y) (u * z)) P = u ^ 2 * F (y - x) (z - x))
  (hPval : eval (vars 1 0 0) P = 4 ∧ eval (vars 0 1 0) P = 5 ∧ eval (vars 0 0 1) P = 6)
  (A : ℂ)
  (B : ℂ)
  (C : ℂ)
  (hPABC : eval (vars A B C) P = 0)
  (habs : ‖B - A‖ = 10)
: (‖C - A‖ = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q111
find_all (answer : ℂ) iff
  (P : MvPolynomial (Fin 3) ℂ)
  (hPreal : ∀ i : Fin 3 →₀ ℕ, (coeff i P).im = 0)
  (F : ℝ → ℝ → ℝ)
  (vars : ℂ → ℂ → ℂ → (Fin 3 → ℂ))
  (hvars : vars = fun a b c ↦ fun i ↦ ite (i = 0) a (ite (i = 1) b c))
  (h : ∀ x y z u : ℝ, eval (vars (u * x) (u * y) (u * z)) P = u ^ 2 * F (y - x) (z - x))
  (hPval : eval (vars 1 0 0) P = 4 ∧ eval (vars 0 1 0) P = 5 ∧ eval (vars 0 0 1) P = 6)
  (A : ℂ)
  (B : ℂ)
  (C : ℂ)
  (hPABC : eval (vars A B C) P = 0)
  (habs : ‖B - A‖ = 10)
: (‖C - A‖ = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
