import FormalProblemSolving.Basic
open Set

/-
## Problem
Determine all polynomials $P(x)$ such that $P(x^2 + 1) = (P(x))^2 + 1$ and $P(0) = 0$.

## Answer
{Polynomial.X}
-/

-- # Formal Answer
noncomputable abbrev Q20.gt := fun (answer : Set (Polynomial ℝ)) => answer = ({(Polynomial.X : Polynomial ℝ)} : Set (Polynomial ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q20
find (answer : Set (Polynomial ℝ)) s.t.
  (P : Polynomial ℝ)
: (∀ P : Polynomial ℝ, P ∈ answer ↔ (P.eval 0 = 0 ∧ (∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q20
find_all (answer : Set (Polynomial ℝ)) iff
  (P : Polynomial ℝ)
: (∀ P : Polynomial ℝ, P ∈ answer ↔ (P.eval 0 = 0 ∧ (∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
