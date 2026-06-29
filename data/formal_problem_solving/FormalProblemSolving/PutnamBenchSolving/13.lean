import FormalProblemSolving.Basic
open Finset Polynomial

/-
## Problem
Let $V$ be the set of all quadratic polynomials with real coefficients such that $|P(x)| \le 1$ for all $x \in [0, 1]$. Find the supremum of $|P'(0)|$ across all $P \in V$.

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q13.gt := fun (answer : ℝ) => answer = (8 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q13
find (answer : ℝ) s.t.
  (V : Set ℝ[X])
  (V_def : V = {P : ℝ[X] | P.degree = 2 ∧ ∀ x ∈ Set.Icc 0 1, |P.eval x| ≤ 1})
: (sSup {|(derivative P).eval 0| | P ∈ V} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q13
find_all (answer : ℝ) iff
  (V : Set ℝ[X])
  (V_def : V = {P : ℝ[X] | P.degree = 2 ∧ ∀ x ∈ Set.Icc 0 1, |P.eval x| ≤ 1})
: (sSup {|(derivative P).eval 0| | P ∈ V} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
