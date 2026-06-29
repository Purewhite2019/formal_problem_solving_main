import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
Let $\mathbf{A}$ and $\mathbf{B}$ be different $n \times n$ matrices with real entries. If $\mathbf{A}^3=\mathbf{B}^3$ and $\mathbf{A}^2\mathbf{B}=\mathbf{B}^2\mathbf{A}$, can $\mathbf{A}^2+\mathbf{B}^2$ be invertible?

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q138.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q138
find (answer : Prop) s.t.
  (n : ℕ)
  (hn : 1 ≤ n)
: (answer ↔ (∃ A B : Matrix (Fin n) (Fin n) ℝ,
    A ≠ B ∧ A ^ 3 = B ^ 3 ∧
    A ^ 2 * B = B ^ 2 * A ∧
    Nonempty (Invertible (A ^ 2 + B ^ 2))))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q138
find_all (answer : Prop) iff
  (n : ℕ)
  (hn : 1 ≤ n)
: (answer ↔ (∃ A B : Matrix (Fin n) (Fin n) ℝ,
    A ≠ B ∧ A ^ 3 = B ^ 3 ∧
    A ^ 2 * B = B ^ 2 * A ∧
    Nonempty (Invertible (A ^ 2 + B ^ 2))))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
