import FormalProblemSolving.Basic
open Polynomial

/-
## Problem
Consider polynomial forms $ax^2-bx+c$ with integer coefficients which have two distinct zeros in the open interval $0<x<1$. Exhibit with a proof the least positive integer value of $a$ for which such a polynomial exists.

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q11.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q11
find (answer : ℕ) s.t.
  (S : Set ℤ)
  (hS : S = {a | ∃ P : Polynomial ℤ, P.degree = 2 ∧ (∃ z1 z2 : Set.Ioo (0 : ℝ) 1, z1 ≠ z2 ∧ aeval (z1 : ℝ) P = 0 ∧ aeval (z2 : ℝ) P = 0) ∧P.coeff 2 = a ∧ a > 0})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q11
find_all (answer : ℕ) iff
  (S : Set ℤ)
  (hS : S = {a | ∃ P : Polynomial ℤ, P.degree = 2 ∧ (∃ z1 z2 : Set.Ioo (0 : ℝ) 1, z1 ≠ z2 ∧ aeval (z1 : ℝ) P = 0 ∧ aeval (z2 : ℝ) P = 0) ∧P.coeff 2 = a ∧ a > 0})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
