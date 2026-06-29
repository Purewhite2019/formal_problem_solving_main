import FormalProblemSolving.Basic

/-
## Problem
Show that if four distinct points of the curve $y = 2x^4 + 7x^3 + 3x - 5$ are collinear, then their average $x$-coordinate is some constant $k$. Find $k$.

## Answer
$-7/8$
-/

-- # Formal Answer
noncomputable abbrev Q45.gt := fun (answer : ℝ) => answer = (-7 / 8 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q45
find (answer : ℝ) s.t.
  (y : ℝ → ℝ)
  (hy : y = fun x ↦ 2 * x ^ 4 + 7 * x ^ 3 + 3 * x - 5)
  (S : Finset ℝ)
  (hS : S.card = 4)
  (h_collinear : Collinear ℝ {P : Fin 2 → ℝ | P 0 ∈ S ∧ P 1 = y (P 0)})
: ((∑ x in S, x) / 4 = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q45
find_all (answer : ℝ) iff
  (y : ℝ → ℝ)
  (hy : y = fun x ↦ 2 * x ^ 4 + 7 * x ^ 3 + 3 * x - 5)
  (S : Finset ℝ)
  (hS : S.card = 4)
  (h_collinear : Collinear ℝ {P : Fin 2 → ℝ | P 0 ∈ S ∧ P 1 = y (P 0)})
: ((∑ x in S, x) / 4 = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
