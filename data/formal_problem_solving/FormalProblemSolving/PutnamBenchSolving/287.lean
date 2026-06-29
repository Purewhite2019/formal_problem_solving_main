import FormalProblemSolving.Basic
open Topology Filter Set Matrix

/-
## Problem
Let $\mathcal{F}$ be the set of functions $f(x,y)$ that are twice continuously differentiable for $x \geq 1,y \geq 1$ and that satisfy the following two equations (where subscripts denote partial derivatives):
\begin{gather*}
xf_x+yf_y=xy\ln(xy), \\
x^2f_{xx}+y^2f_{yy}=xy.
\end{gather*}
For each $f \in \mathcal{F}$, let $m(f)=\min_{s \geq 1} (f(s+1,s+1)-f(s+1,s)-f(s,s+1)+f(s,s))$. Determine $m(f)$, and show that it is independent of the choice of $f$.

## Answer
$2 \ln 2 - \frac{1}{2}$
-/

-- # Formal Answer
noncomputable abbrev Q287.gt := fun (answer : ℝ) => answer = (((2 : ℝ) * (Real.log (2 : ℝ) : ℝ) : ℝ) - (1 / 2 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q287
find (answer : ℝ) s.t.
  (f : (Fin 2 → ℝ) → ℝ)
  (vec : ℝ → ℝ → (Fin 2 → ℝ))
  (fdiff : ContDiff ℝ 2 f)
  (hvec : ∀ x y : ℝ, (vec x y) 0 = x ∧ (vec x y 1) = y)
  (feq1 : ∀ x ≥ 1, ∀ y ≥ 1, x * deriv (fun x' : ℝ => f (vec x' y)) x + y * deriv (fun y' : ℝ => f (vec x y')) y = x * y * Real.log (x * y))
  (feq2 : ∀ x ≥ 1, ∀ y ≥ 1, x ^ 2 * iteratedDeriv 2 (fun x' : ℝ => f (vec x' y)) x + y ^ 2 * iteratedDeriv 2 (fun y' : ℝ => f (vec x y')) y = x * y)
: (sInf {f (vec (s + 1) (s + 1)) - f (vec (s + 1) s) - f (vec s (s + 1)) + f (vec s s) | s ≥ 1} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q287
find_all (answer : ℝ) iff
  (f : (Fin 2 → ℝ) → ℝ)
  (vec : ℝ → ℝ → (Fin 2 → ℝ))
  (fdiff : ContDiff ℝ 2 f)
  (hvec : ∀ x y : ℝ, (vec x y) 0 = x ∧ (vec x y 1) = y)
  (feq1 : ∀ x ≥ 1, ∀ y ≥ 1, x * deriv (fun x' : ℝ => f (vec x' y)) x + y * deriv (fun y' : ℝ => f (vec x y')) y = x * y * Real.log (x * y))
  (feq2 : ∀ x ≥ 1, ∀ y ≥ 1, x ^ 2 * iteratedDeriv 2 (fun x' : ℝ => f (vec x' y)) x + y ^ 2 * iteratedDeriv 2 (fun y' : ℝ => f (vec x y')) y = x * y)
: (sInf {f (vec (s + 1) (s + 1)) - f (vec (s + 1) s) - f (vec s (s + 1)) + f (vec s s) | s ≥ 1} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
