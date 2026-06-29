import FormalProblemSolving.Basic
open Real Equiv Polynomial Filter Topology MvPolynomial

/-
## Problem
Let $f(x,y,z) = x^2+y^2+z^2+xyz$. Let $p(x,y,z), q(x,y,z)$, $r(x,y,z)$ be polynomials with real coefficients satisfying
\[
f(p(x,y,z), q(x,y,z), r(x,y,z)) = f(x,y,z).
\]
Prove or disprove the assertion that the sequence $p,q,r$ consists of some permutation of $\pm x, \pm y, \pm z$, where the number of minus signs is $0$ or $2$.

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q109.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q109
find (answer : Prop) s.t.
  (f : MvPolynomial (Fin 3) ℝ)
  (perms : Set (Set (MvPolynomial (Fin 3) ℝ)))
  (hf : f = (X 0) ^ 2 + (X 1) ^ 2 + (X 2) ^ 2 + (X 0) * (X 1) * (X 2))
  (hperms : perms = {{X 0, X 1, X 2}, {X 0, -X 1, -X 2}, {-X 0, X 1, -X 2}, {-X 0, -X 1, X 2}})
: (answer ↔ (∀ pqr : Fin 3 → MvPolynomial (Fin 3) ℝ,
      (∀ xyz : Fin 3 → ℝ, MvPolynomial.eval (fun i ↦ MvPolynomial.eval xyz (pqr i)) f = MvPolynomial.eval xyz f) →
      ({pqr 0, pqr 1, pqr 2} ∈ perms)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q109
find_all (answer : Prop) iff
  (f : MvPolynomial (Fin 3) ℝ)
  (perms : Set (Set (MvPolynomial (Fin 3) ℝ)))
  (hf : f = (X 0) ^ 2 + (X 1) ^ 2 + (X 2) ^ 2 + (X 0) * (X 1) * (X 2))
  (hperms : perms = {{X 0, X 1, X 2}, {X 0, -X 1, -X 2}, {-X 0, X 1, -X 2}, {-X 0, -X 1, X 2}})
: (answer ↔ (∀ pqr : Fin 3 → MvPolynomial (Fin 3) ℝ,
      (∀ xyz : Fin 3 → ℝ, MvPolynomial.eval (fun i ↦ MvPolynomial.eval xyz (pqr i)) f = MvPolynomial.eval xyz f) →
      ({pqr 0, pqr 1, pqr 2} ∈ perms)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
