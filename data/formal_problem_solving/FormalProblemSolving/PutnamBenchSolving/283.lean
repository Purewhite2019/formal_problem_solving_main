import FormalProblemSolving.Basic
open MeasureTheory Metric Topology Filter

/-
## Problem
Let $f$ be a continuous real-valued function on $\mathbb{R}^3$. Suppose that for every sphere $S$ of radius $1$, the integral of $f(x,y,z)$ over the surface of $S$ equals $0$. Must $f(x,y,z)$ be identically 0?

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q283.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q283
find (answer : Prop) s.t.
  (P : (EuclideanSpace ℝ (Fin 3) → ℝ) → Prop)
  (P_def : ∀ f, P f ↔ ∀ C, ∫ x in sphere C 1, f x ∂μH[2] = 0)
: ((∀ f, Continuous f → P f → f = 0) ↔ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q283
find_all (answer : Prop) iff
  (P : (EuclideanSpace ℝ (Fin 3) → ℝ) → Prop)
  (P_def : ∀ f, P f ↔ ∀ C, ∫ x in sphere C 1, f x ∂μH[2] = 0)
: ((∀ f, Continuous f → P f → f = 0) ↔ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
