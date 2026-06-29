import FormalProblemSolving.Basic

/-
## Problem
Alice and Bob play a game on a board consisting of one row of 2022 consecutive squares. They take turns placing tiles that cover two adjacent squares, with Alice going first. By rule, a tile must not cover a square that is already covered by another tile. The game ends when no tile can be placed according to this rule. Alice's goal is to maximize the number of uncovered squares when the game ends; Bob's goal is to minimize it. What is the greatest number of uncovered squares that Alice can ensure at the end of the game, no matter how Bob plays?

## Answer
290
-/

-- # Formal Answer
noncomputable abbrev Q307.gt := fun (answer : ℕ) => answer = (290 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q307
find (answer : ℕ) s.t.
  (IsValidMove : Set (Fin 2022) → Set (Fin 2022) → Prop)
  (IsValidMove_def : ∀ x y, IsValidMove x y ↔
      (x = y ∧ ∀ i < 2021, i ∉ x → i + 1 ∈ x) ∨
      ∃ i < 2021, i ∉ x ∧ i + 1 ∉ x ∧ y = x ∪ {i, i + 1})
  (IsValidGame : List (Set (Fin 2022)) → Prop)
  (IsValidGame_def : ∀ g, IsValidGame g ↔ (∃ gt, g = ∅ :: gt) ∧ g.Chain' IsValidMove)
  (ConformsToStrategy : List (Set (Fin 2022)) → (Set (Fin 2022) → Set (Fin 2022)) → Prop)
  (ConformsToStrategy_def : ∀ g s, ConformsToStrategy g s ↔
      ∀ (i) (h : i + 1 < g.length), Even i → g[i + 1] = s g[i])
: (IsGreatest
      {n | ∃ s, (∀ x, IsValidMove x (s x)) ∧ ∀ g,
        IsValidGame g → ConformsToStrategy g s → ∃ gh x, g = gh ++ [x] ∧ n ≤ xᶜ.ncard}
      answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q307
find_all (answer : ℕ) iff
  (IsValidMove : Set (Fin 2022) → Set (Fin 2022) → Prop)
  (IsValidMove_def : ∀ x y, IsValidMove x y ↔
      (x = y ∧ ∀ i < 2021, i ∉ x → i + 1 ∈ x) ∨
      ∃ i < 2021, i ∉ x ∧ i + 1 ∉ x ∧ y = x ∪ {i, i + 1})
  (IsValidGame : List (Set (Fin 2022)) → Prop)
  (IsValidGame_def : ∀ g, IsValidGame g ↔ (∃ gt, g = ∅ :: gt) ∧ g.Chain' IsValidMove)
  (ConformsToStrategy : List (Set (Fin 2022)) → (Set (Fin 2022) → Set (Fin 2022)) → Prop)
  (ConformsToStrategy_def : ∀ g s, ConformsToStrategy g s ↔
      ∀ (i) (h : i + 1 < g.length), Even i → g[i + 1] = s g[i])
: (IsGreatest
      {n | ∃ s, (∀ x, IsValidMove x (s x)) ∧ ∀ g,
        IsValidGame g → ConformsToStrategy g s → ∃ gh x, g = gh ++ [x] ∧ n ≤ xᶜ.ncard}
      answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
