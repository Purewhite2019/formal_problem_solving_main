import FormalProblemSolving.Basic
open Set

/-
## Problem
Let $A$ be a set of $2n$ points in the plane, $n$ colored red and $n$ colored blue, such that no three points in $A$ are collinear. Must there exist $n$ closed straight line segments, each connecting one red and one blue point in $A$, such that no two of the $n$ line segments intersect?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q58.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q58
find (answer : Prop) s.t.
  (A : Finset (Fin 2 → ℝ) × Finset (Fin 2 → ℝ) → Prop)
  (hA : A = fun (R, B) => R.card = B.card ∧ R ∩ B = ∅ ∧
    ∀ u : Finset (Fin 2 → ℝ), u ⊆ R ∪ B → u.card = 3 → ¬Collinear ℝ (u : Set (Fin 2 → ℝ)))
  (w : (Fin 2 → ℝ) × (Fin 2 → ℝ) → ℝ → (Fin 2 → ℝ))
  (hw : w = fun (P, Q) => fun x : ℝ => fun i : Fin 2 => x * P i + (1 - x) * Q i)
: (answer ↔
      (∀ R B, A (R, B) →
        ∃ v : Finset ((Fin 2 → ℝ) × (Fin 2 → ℝ)),
          (∀ L ∈ v, ∀ M ∈ v, L ≠ M → ∀ x ∈ Icc 0 1, ∀ y ∈ Icc 0 1,
          Real.sqrt ((w (L.1, L.2) x 0 - w (M.1, M.2) y 0)^2 + (w (L.1, L.2) x 1 - w (M.1, M.2) y 1)^2) ≠ 0) ∧
        v.card = R.card ∧ ∀ L ∈ v, L.1 ∈ R ∧ L.2 ∈ B))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q58
find_all (answer : Prop) iff
  (A : Finset (Fin 2 → ℝ) × Finset (Fin 2 → ℝ) → Prop)
  (hA : A = fun (R, B) => R.card = B.card ∧ R ∩ B = ∅ ∧
    ∀ u : Finset (Fin 2 → ℝ), u ⊆ R ∪ B → u.card = 3 → ¬Collinear ℝ (u : Set (Fin 2 → ℝ)))
  (w : (Fin 2 → ℝ) × (Fin 2 → ℝ) → ℝ → (Fin 2 → ℝ))
  (hw : w = fun (P, Q) => fun x : ℝ => fun i : Fin 2 => x * P i + (1 - x) * Q i)
: (answer ↔
      (∀ R B, A (R, B) →
        ∃ v : Finset ((Fin 2 → ℝ) × (Fin 2 → ℝ)),
          (∀ L ∈ v, ∀ M ∈ v, L ≠ M → ∀ x ∈ Icc 0 1, ∀ y ∈ Icc 0 1,
          Real.sqrt ((w (L.1, L.2) x 0 - w (M.1, M.2) y 0)^2 + (w (L.1, L.2) x 1 - w (M.1, M.2) y 1)^2) ≠ 0) ∧
        v.card = R.card ∧ ∀ L ∈ v, L.1 ∈ R ∧ L.2 ∈ B))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
