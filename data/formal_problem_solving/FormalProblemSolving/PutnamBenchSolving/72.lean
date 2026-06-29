import FormalProblemSolving.Basic
open Topology Filter Set Polynomial Function

/-
## Problem
Let $V$ be a set of $5$ by $7$ matrices, with real entries and with the property that $rA+sB \in V$ whenever $A,B \in V$ and $r$ and $s$ are scalars (i.e., real numbers). \emph{Prove or disprove} the following assertion: If $V$ contains matrices of ranks $0$, $1$, $2$, $4$, and $5$, then it also contains a matrix of rank $3$. [The rank of a nonzero matrix $M$ is the largest $k$ such that the entries of some $k$ rows and some $k$ columns form a $k$ by $k$ matrix with a nonzero determinant.]

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q72.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q72
find (answer : Prop) s.t.
  (VAB : Set (Matrix (Fin 5) (Fin 7) ℝ) → Prop)
  (Vrank : Set (Matrix (Fin 5) (Fin 7) ℝ) → ℕ → Prop)
  (hVAB : ∀ V, VAB V = (∀ A ∈ V, ∀ B ∈ V, ∀ r s : ℝ, r • A + s • B ∈ V))
  (hVrank : ∀ V k, Vrank V k = ∃ A ∈ V, A.rank = k)
: (answer ↔
    (∀ V,
      VAB V → Vrank V 0 → Vrank V 1 →
      Vrank V 2 → Vrank V 4 → Vrank V 5 →
    Vrank V 3))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q72
find_all (answer : Prop) iff
  (VAB : Set (Matrix (Fin 5) (Fin 7) ℝ) → Prop)
  (Vrank : Set (Matrix (Fin 5) (Fin 7) ℝ) → ℕ → Prop)
  (hVAB : ∀ V, VAB V = (∀ A ∈ V, ∀ B ∈ V, ∀ r s : ℝ, r • A + s • B ∈ V))
  (hVrank : ∀ V k, Vrank V k = ∃ A ∈ V, A.rank = k)
: (answer ↔
    (∀ V,
      VAB V → Vrank V 0 → Vrank V 1 →
      Vrank V 2 → Vrank V 4 → Vrank V 5 →
    Vrank V 3))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
