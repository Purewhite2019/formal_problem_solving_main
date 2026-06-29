import FormalProblemSolving.Basic
open Filter Topology Nat

/-
## Problem
If $\mathbf{A}$ and $\mathbf{B}$ are square matrices of the same size such that $\mathbf{ABAB}=\mathbf{0}$, does it follow that $\mathbf{BABA}=\mathbf{0}$?

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q133.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q133
find (answer : Prop) s.t.
: (answer ↔
    (∀ n ≥ 1, ∀ A B : Matrix (Fin n) (Fin n) ℝ,
      A * B * A * B = 0 → B * A * B * A = 0))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q133
find_all (answer : Prop) iff
: (answer ↔
    (∀ n ≥ 1, ∀ A B : Matrix (Fin n) (Fin n) ℝ,
      A * B * A * B = 0 → B * A * B * A = 0))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
