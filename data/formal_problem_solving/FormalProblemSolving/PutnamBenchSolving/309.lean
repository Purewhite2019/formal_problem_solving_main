import FormalProblemSolving.Basic
open Polynomial

/-
## Problem
Let $\times$ represent the cross product in $\mathbb{R}^3$. For what positive integers $n$ does there exist a set $S \subset \mathbb{R}^3$ with exactly $n$ elements such that $S=\{v \times w:v,w \in S\}$?

## Answer
{1, 7}
-/

-- # Formal Answer
noncomputable abbrev Q309.gt := fun (answer : Set ℕ) => answer = (Insert.insert (1 : ℕ) ({(7 : ℕ)} : Set ℕ) : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q309
find (answer : Set ℕ) s.t.
  (n : ℕ)
  (P : Finset (Fin 3 → ℝ) → Prop)
  (P_def : ∀ S : Finset (Fin 3 → ℝ), P S ↔ (S = {u : Fin 3 → ℝ | ∃ v w : S, u = crossProduct v w}))
: ((0 < n ∧ ∃ S : Finset (Fin 3 → ℝ), S.card = n ∧ P S) ↔ n ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q309
find_all (answer : Set ℕ) iff
  (n : ℕ)
  (P : Finset (Fin 3 → ℝ) → Prop)
  (P_def : ∀ S : Finset (Fin 3 → ℝ), P S ↔ (S = {u : Fin 3 → ℝ | ∃ v w : S, u = crossProduct v w}))
: ((0 < n ∧ ∃ S : Finset (Fin 3 → ℝ), S.card = n ∧ P S) ↔ n ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
