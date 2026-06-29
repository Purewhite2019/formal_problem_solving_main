import FormalProblemSolving.Basic
open MeasureTheory

/-
## Problem
Find the least possible area of a convex set in the plane that intersects both branches of the hyperbola $xy=1$ and both branches of the hyperbola $xy=-1$. (A set $S$ in the plane is called \emph{convex} if for any two points in $S$ the line segment connecting them is contained in $S$.)

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q211.gt := fun (answer : ENNReal) => answer = (4 : ENNReal)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q211
find (answer : ENNReal) s.t.
: (IsLeast
    {y | ∃ S : Set (Fin 2 → ℝ),
      Convex ℝ S ∧
      (∃ p ∈ S, p 0 > 0 ∧ p 1 > 0 ∧ p 0 * p 1 = 1) ∧
      (∃ p ∈ S, p 0 < 0 ∧ p 1 < 0 ∧ p 0 * p 1 = 1) ∧
      (∃ p ∈ S, p 0 < 0 ∧ p 1 > 0 ∧ p 0 * p 1 = -1) ∧
      (∃ p ∈ S, p 0 > 0 ∧ p 1 < 0 ∧ p 0 * p 1 = -1) ∧
    volume S = y} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q211
find_all (answer : ENNReal) iff
: (IsLeast
    {y | ∃ S : Set (Fin 2 → ℝ),
      Convex ℝ S ∧
      (∃ p ∈ S, p 0 > 0 ∧ p 1 > 0 ∧ p 0 * p 1 = 1) ∧
      (∃ p ∈ S, p 0 < 0 ∧ p 1 < 0 ∧ p 0 * p 1 = 1) ∧
      (∃ p ∈ S, p 0 < 0 ∧ p 1 > 0 ∧ p 0 * p 1 = -1) ∧
      (∃ p ∈ S, p 0 > 0 ∧ p 1 < 0 ∧ p 0 * p 1 = -1) ∧
    volume S = y} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
