import FormalProblemSolving.Basic
open Topology Filter

/-
## Problem
Define $C(\alpha)$ to be the coefficient of $x^{1992}$ in the power series about $x=0$ of $(1 + x)^\alpha$. Evaluate
\[
\int_0^1 \left( C(-y-1) \sum_{k=1}^{1992} \frac{1}{y+k} \right)\,dy.
\]

## Answer
1992
-/

-- # Formal Answer
noncomputable abbrev Q145.gt := fun (answer : ℝ) => answer = (1992 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q145
find (answer : ℝ) s.t.
  (C : ℝ → ℝ)
  (hC : C = fun α ↦ taylorCoeffWithin (fun x ↦ (1 + x) ^ α) 1992 Set.univ 0)
: (∫ y in (0)..1, C (-y - 1) * ∑ k in Finset.Icc (1 : ℕ) 1992, 1 / (y + k) = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q145
find_all (answer : ℝ) iff
  (C : ℝ → ℝ)
  (hC : C = fun α ↦ taylorCoeffWithin (fun x ↦ (1 + x) ^ α) 1992 Set.univ 0)
: (∫ y in (0)..1, C (-y - 1) * ∑ k in Finset.Icc (1 : ℕ) 1992, 1 / (y + k) = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
