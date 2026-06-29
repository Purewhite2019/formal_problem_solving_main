import FormalProblemSolving.Basic
open Set Filter Topology

/-
## Problem
\begin{enumerate}
\item[(a)] If every point of the plane is painted one of three colors, do there necessarily exist two points of the same color exactly one inch apart?
\item[(b)] What if ``three'' is replaced by ``nine''?
\end{enumerate}

## Answer
(True, False)
-/

-- # Formal Answer
noncomputable abbrev Q119.gt := fun (answer : Prop × Prop) => answer = ((True, False) : Prop × Prop)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q119
find (answer : Prop × Prop) s.t.
  (p : ℕ → Prop)
  (hp : ∀ n, p n ↔
      ∀ color : (EuclideanSpace ℝ (Fin 2)) → Fin n,
        ∃ p q : EuclideanSpace ℝ (Fin 2),
          color p = color q ∧ dist p q = 1)
: (let (a, b) := answer; (p 3 ↔ a) ∧ (p 9 ↔ b))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q119
find_all (answer : Prop × Prop) iff
  (p : ℕ → Prop)
  (hp : ∀ n, p n ↔
      ∀ color : (EuclideanSpace ℝ (Fin 2)) → Fin n,
        ∃ p q : EuclideanSpace ℝ (Fin 2),
          color p = color q ∧ dist p q = 1)
: (let (a, b) := answer; (p 3 ↔ a) ∧ (p 9 ↔ b))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
