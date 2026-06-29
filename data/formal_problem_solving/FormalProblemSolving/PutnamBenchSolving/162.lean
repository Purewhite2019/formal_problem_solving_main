import FormalProblemSolving.Basic
open Metric

/-
## Problem
Let $C_1$ and $C_2$ be circles whose centers are $10$ units apart, and whose radii are $1$ and $3$. Find, with proof, the locus of all points $M$ for which there exist points $X$ on $C_1$ and $Y$ on $C_2$ such that $M$ is the midpoint of the line segment $XY$.

## Answer
the set of points $p$ such that the distance from $p$ to the midpoint of $O_1$ and $O_2$ is between 1 and 2
-/

-- # Formal Answer
noncomputable abbrev Q162.gt := fun (O1 : EuclideanSpace ℝ (Fin 2)) (O2 : EuclideanSpace ℝ (Fin 2)) (answer : (EuclideanSpace ℝ (Fin 2)) → (EuclideanSpace ℝ (Fin 2)) → Set (EuclideanSpace ℝ (Fin 2))) => answer = fun (O1 O2 : EuclideanSpace ℝ (Fin (2 : ℕ))) =>
  ({p : EuclideanSpace ℝ (Fin (2 : ℕ)) |
      (Dist.dist p (midpoint ℝ O1 O2 : EuclideanSpace ℝ (Fin (2 : ℕ))) : ℝ) ≥ (1 : ℝ) ∧
        (Dist.dist p (midpoint ℝ O1 O2 : EuclideanSpace ℝ (Fin (2 : ℕ))) : ℝ) ≤ (2 : ℝ)} :
    Set (EuclideanSpace ℝ (Fin (2 : ℕ))))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q162
  (O1 : EuclideanSpace ℝ (Fin 2))
  (O2 : EuclideanSpace ℝ (Fin 2))
find (answer : (EuclideanSpace ℝ (Fin 2)) → (EuclideanSpace ℝ (Fin 2)) → Set (EuclideanSpace ℝ (Fin 2))) s.t.
  (C1 : Set (EuclideanSpace ℝ (Fin 2)))
  (C2 : Set (EuclideanSpace ℝ (Fin 2)))
  (hC1 : C1 = Metric.sphere O1 1)
  (hC2 : C2 = Metric.sphere O2 3)
  (hO1O2 : dist O1 O2 = 10)
: ({M : EuclideanSpace ℝ (Fin 2) | ∃ X Y, X ∈ C1 ∧ Y ∈ C2 ∧ M = midpoint ℝ X Y} = answer O1 O2)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q162
  (O1 : EuclideanSpace ℝ (Fin 2))
  (O2 : EuclideanSpace ℝ (Fin 2))
find_all (answer : (EuclideanSpace ℝ (Fin 2)) → (EuclideanSpace ℝ (Fin 2)) → Set (EuclideanSpace ℝ (Fin 2))) iff
  (C1 : Set (EuclideanSpace ℝ (Fin 2)))
  (C2 : Set (EuclideanSpace ℝ (Fin 2)))
  (hC1 : C1 = Metric.sphere O1 1)
  (hC2 : C2 = Metric.sphere O2 3)
  (hO1O2 : dist O1 O2 = 10)
: ({M : EuclideanSpace ℝ (Fin 2) | ∃ X Y, X ∈ C1 ∧ Y ∈ C2 ∧ M = midpoint ℝ X Y} = answer O1 O2)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
