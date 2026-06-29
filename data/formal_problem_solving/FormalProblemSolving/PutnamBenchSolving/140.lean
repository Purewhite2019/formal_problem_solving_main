import FormalProblemSolving.Basic
open Filter Metric Topology

/-
## Problem
Does there exist an infinite sequence of closed discs $D_1,D_2,D_3,\dots$ in the plane, with centers $c_1,c_2,c_3,\dots$, respectively, such that
\begin{enumerate}
\item the $c_i$ have no limit point in the finite plane,
\item the sum of the areas of the $D_i$ is finite, and
\item every line in the plane intersects at least one of the $D_i$?
\end{enumerate}

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q140.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q140
find (answer : Prop) s.t.
: ((∃ (c : ℕ → EuclideanSpace ℝ (Fin 2)) (r : ℕ → ℝ),
      (¬ ∃ p, MapClusterPt p atTop c) ∧
      (Summable <| fun i ↦ r i ^ 2) ∧
      (∀ L : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2)),
        Module.finrank ℝ L.direction = 1 → ∃ i, (↑L ∩ closedBall (c i) (r i)).Nonempty)) ↔
    answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q140
find_all (answer : Prop) iff
: ((∃ (c : ℕ → EuclideanSpace ℝ (Fin 2)) (r : ℕ → ℝ),
      (¬ ∃ p, MapClusterPt p atTop c) ∧
      (Summable <| fun i ↦ r i ^ 2) ∧
      (∀ L : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2)),
        Module.finrank ℝ L.direction = 1 → ∃ i, (↑L ∩ closedBall (c i) (r i)).Nonempty)) ↔
    answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
