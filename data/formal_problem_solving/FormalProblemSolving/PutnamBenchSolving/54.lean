import FormalProblemSolving.Basic
open Set Real Filter Topology Polynomial

/-
## Problem
Find the real polynomial $p(x)$ of degree $4$ with largest possible coefficient of $x^4$ such that $p([-1, 1]) \subseteq [0, 1]$.

## Answer
$4x^4 - 4x^2 + 1$
-/

-- # Formal Answer
noncomputable abbrev Q54.gt := fun (answer : Polynomial ℝ) => answer =
  ((((4 : ℝ[X]) * ((Polynomial.X : ℝ[X]) ^ (4 : ℕ) : ℝ[X]) : ℝ[X]) -
          ((4 : ℝ[X]) * ((Polynomial.X : ℝ[X]) ^ (2 : ℕ) : ℝ[X]) : ℝ[X]) :
        ℝ[X]) +
      (1 : ℝ[X]) :
    ℝ[X])

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q54
find (answer : Polynomial ℝ) s.t.
  (S : Set (Polynomial ℝ))
  (hS : S = {p : Polynomial ℝ | p.degree = 4 ∧ ∀ x ∈ Icc (-1 : ℝ) 1, p.eval x ∈ Icc 0 1})
: (answer ∈ S ∧ (∀ p ∈ S, p.coeff 4 ≤ answer.coeff 4))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q54
find_all (answer : Polynomial ℝ) iff
  (S : Set (Polynomial ℝ))
  (hS : S = {p : Polynomial ℝ | p.degree = 4 ∧ ∀ x ∈ Icc (-1 : ℝ) 1, p.eval x ∈ Icc 0 1})
: (answer ∈ S ∧ (∀ p ∈ S, p.coeff 4 ≤ answer.coeff 4))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
