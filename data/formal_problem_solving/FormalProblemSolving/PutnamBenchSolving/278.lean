import FormalProblemSolving.Basic

/-
## Problem
Determine the greatest possible value of $\sum_{i=1}^{10} \cos(3x_i)$ for real numbers $x_1, x_2, \ldots, x_{10}$ satisfying $\sum_{i=1}^{10} \cos(x_i) = 0$.

## Answer
480/49
-/

-- # Formal Answer
noncomputable abbrev Q278.gt := fun (answer : ℝ) => answer = (480 / 49 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q278
find (answer : ℝ) s.t.
: (IsGreatest
      {∑ i, Real.cos (3 * x i) | (x : Fin 10 → ℝ) (hx : ∑ i, Real.cos (x i) = 0)}
      answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q278
find_all (answer : ℝ) iff
: (IsGreatest
      {∑ i, Real.cos (3 * x i) | (x : Fin 10 → ℝ) (hx : ∑ i, Real.cos (x i) = 0)}
      answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
