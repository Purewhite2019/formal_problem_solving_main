import FormalProblemSolving.Basic
open Finset Polynomial

/-
## Problem
Find all polynomials of the form $\sum_{0}^{n} a_{i} x^{n-i}$ with $n \ge 1$ and $a_i = \pm 1$ for all $0 \le i \le n$ whose roots are all real.

## Answer
{X - 1, -(X - 1), X + 1, -(X + 1), X^2 + X - 1, -(X^2 + X - 1), X^2 - X - 1, -(X^2 - X - 1), X^3 + X^2 - X - 1, -(X^3 + X^2 - X - 1), X^3 - X^2 - X + 1, -(X^3 - X^2 - X + 1)}
-/

-- # Formal Answer
noncomputable abbrev Q14.gt := fun (answer : Set ℂ[X]) => answer =
  (Insert.insert ((Polynomial.X : ℂ[X]) - (1 : ℂ[X]) : ℂ[X])
      (Insert.insert (-((Polynomial.X : ℂ[X]) - (1 : ℂ[X]) : ℂ[X]) : ℂ[X])
          (Insert.insert ((Polynomial.X : ℂ[X]) + (1 : ℂ[X]) : ℂ[X])
              (Insert.insert (-((Polynomial.X : ℂ[X]) + (1 : ℂ[X]) : ℂ[X]) : ℂ[X])
                  (Insert.insert
                      ((((Polynomial.X : ℂ[X]) ^ (2 : ℕ) : ℂ[X]) + (Polynomial.X : ℂ[X]) : ℂ[X]) - (1 : ℂ[X]) : ℂ[X])
                      (Insert.insert
                          (-((((Polynomial.X : ℂ[X]) ^ (2 : ℕ) : ℂ[X]) + (Polynomial.X : ℂ[X]) : ℂ[X]) - (1 : ℂ[X]) :
                                ℂ[X]) :
                            ℂ[X])
                          (Insert.insert
                              ((((Polynomial.X : ℂ[X]) ^ (2 : ℕ) : ℂ[X]) - (Polynomial.X : ℂ[X]) : ℂ[X]) - (1 : ℂ[X]) :
                                ℂ[X])
                              (Insert.insert
                                  (-((((Polynomial.X : ℂ[X]) ^ (2 : ℕ) : ℂ[X]) - (Polynomial.X : ℂ[X]) : ℂ[X]) -
                                          (1 : ℂ[X]) :
                                        ℂ[X]) :
                                    ℂ[X])
                                  (Insert.insert
                                      (((((Polynomial.X : ℂ[X]) ^ (3 : ℕ) : ℂ[X]) +
                                                  ((Polynomial.X : ℂ[X]) ^ (2 : ℕ) : ℂ[X]) :
                                                ℂ[X]) -
                                              (Polynomial.X : ℂ[X]) :
                                            ℂ[X]) -
                                          (1 : ℂ[X]) :
                                        ℂ[X])
                                      (Insert.insert
                                          (-(((((Polynomial.X : ℂ[X]) ^ (3 : ℕ) : ℂ[X]) +
                                                          ((Polynomial.X : ℂ[X]) ^ (2 : ℕ) : ℂ[X]) :
                                                        ℂ[X]) -
                                                      (Polynomial.X : ℂ[X]) :
                                                    ℂ[X]) -
                                                  (1 : ℂ[X]) :
                                                ℂ[X]) :
                                            ℂ[X])
                                          (Insert.insert
                                              (((((Polynomial.X : ℂ[X]) ^ (3 : ℕ) : ℂ[X]) -
                                                          ((Polynomial.X : ℂ[X]) ^ (2 : ℕ) : ℂ[X]) :
                                                        ℂ[X]) -
                                                      (Polynomial.X : ℂ[X]) :
                                                    ℂ[X]) +
                                                  (1 : ℂ[X]) :
                                                ℂ[X])
                                              ({(-(((((Polynomial.X : ℂ[X]) ^ (3 : ℕ) : ℂ[X]) -
                                                                  ((Polynomial.X : ℂ[X]) ^ (2 : ℕ) : ℂ[X]) :
                                                                ℂ[X]) -
                                                              (Polynomial.X : ℂ[X]) :
                                                            ℂ[X]) +
                                                          (1 : ℂ[X]) :
                                                        ℂ[X]) :
                                                    ℂ[X])} :
                                                Set ℂ[X]) :
                                            Set ℂ[X]) :
                                        Set ℂ[X]) :
                                    Set ℂ[X]) :
                                Set ℂ[X]) :
                            Set ℂ[X]) :
                        Set ℂ[X]) :
                    Set ℂ[X]) :
                Set ℂ[X]) :
            Set ℂ[X]) :
        Set ℂ[X]) :
    Set ℂ[X])

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q14
find (answer : Set ℂ[X]) s.t.
: ({P : ℂ[X] | P.natDegree ≥ 1 ∧ (∀ k ∈ Set.Icc 0 P.natDegree, P.coeff k = 1 ∨ P.coeff k = -1) ∧
    ∀ z : ℂ, P.eval z = 0 → ∃ r : ℝ, r = z} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q14
find_all (answer : Set ℂ[X]) iff
: ({P : ℂ[X] | P.natDegree ≥ 1 ∧ (∀ k ∈ Set.Icc 0 P.natDegree, P.coeff k = 1 ∨ P.coeff k = -1) ∧
    ∀ z : ℂ, P.eval z = 0 → ∃ r : ℝ, r = z} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
