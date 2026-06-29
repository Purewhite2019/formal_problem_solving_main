import FormalProblemSolving.Basic
open Matrix Filter Topology Set Nat

/-
## Problem
Show that a finite group can not be the union of two of its proper subgroups. Does the statement remain true if 'two' is replaced by 'three'?

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q17.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q17
find (answer : Prop) s.t.
  (P : ℕ → Prop)
  (P_def : ∀ n, P n ↔ ∀ (G : Type) [Group G] [Finite G],
      ∀ H : Fin n → Subgroup G, (∀ i, H i < ⊤) → ⋃ i, (H i : Set G) < ⊤)
: (answer ↔ (P 3))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q17
find_all (answer : Prop) iff
  (P : ℕ → Prop)
  (P_def : ∀ n, P n ↔ ∀ (G : Type) [Group G] [Finite G],
      ∀ H : Fin n → Subgroup G, (∀ i, H i < ⊤) → ⋃ i, (H i : Set G) < ⊤)
: (answer ↔ (P 3))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
