import FormalProblemSolving.Basic
open EuclideanGeometry Real

/-
## Problem
Let $\triangle ABC$ satisfy $\angle CAB < \angle BCA < \frac{\pi}{2} < \angle ABC$. If the bisector of the external angle at $A$ meets line $BC$ at $P$, the bisector of the external angle at $B$ meets line $CA$ at $Q$, and $AP = BQ = AB$, find $\angle CAB$.

## Answer
π / 15
-/

-- # Formal Answer
noncomputable abbrev Q8.gt := fun (answer : ℝ) => answer = ((π : ℝ) / (15 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q8
find (answer : ℝ) s.t.
  (A : EuclideanSpace ℝ (Fin 2))
  (B : EuclideanSpace ℝ (Fin 2))
  (C : EuclideanSpace ℝ (Fin 2))
  (X : EuclideanSpace ℝ (Fin 2))
  (Y : EuclideanSpace ℝ (Fin 2))
  (hABC : ¬Collinear ℝ {A, B, C})
  (hangles : ∠ C A B < ∠ B C A ∧ ∠ B C A < π/2 ∧ π/2 < ∠ A B C)
  (hX : Collinear ℝ {X, B, C} ∧ ∠ X A B = (π - ∠ C A B)/2 ∧ dist A X = dist A B)
  (hY : Collinear ℝ {Y, C, A} ∧ ∠ Y B C = (π - ∠ A B C)/2 ∧ dist B Y = dist A B)
: (∠ C A B = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q8
find_all (answer : ℝ) iff
  (A : EuclideanSpace ℝ (Fin 2))
  (B : EuclideanSpace ℝ (Fin 2))
  (C : EuclideanSpace ℝ (Fin 2))
  (X : EuclideanSpace ℝ (Fin 2))
  (Y : EuclideanSpace ℝ (Fin 2))
  (hABC : ¬Collinear ℝ {A, B, C})
  (hangles : ∠ C A B < ∠ B C A ∧ ∠ B C A < π/2 ∧ π/2 < ∠ A B C)
  (hX : Collinear ℝ {X, B, C} ∧ ∠ X A B = (π - ∠ C A B)/2 ∧ dist A X = dist A B)
  (hY : Collinear ℝ {Y, C, A} ∧ ∠ Y B C = (π - ∠ A B C)/2 ∧ dist B Y = dist A B)
: (∠ C A B = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
