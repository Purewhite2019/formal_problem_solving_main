import FormalProblemSolving.Basic

/-
## Problem
Let $S$ be the smallest set of positive integers such that (a) $2$ is in $S$, (b) $n$ is in $S$ whenever $n^2$ is in $S$, and (c) $(n+5)^2$ is in $S$ whenever $n$ is in $S$. Which positive integers are not in $S$?.

## Answer
{x : ℤ | x > 0 ∧ (x = 1 ∨ 5 ∣ x)}
-/

-- # Formal Answer
noncomputable abbrev Q272.gt := fun (answer : Set ℤ) => answer = ({x : ℤ | x > (0 : ℤ) ∧ (x = (1 : ℤ) ∨ (5 : ℤ) ∣ x)} : Set ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q272
find (answer : Set ℤ) s.t.
  (IsQualifying : Set ℤ → Prop)
  (IsQualifying_def : ∀ S, IsQualifying S ↔
  (∀ n ∈ S, 0 < n) ∧
  2 ∈ S ∧
  (∀ n > 0, n ^ 2 ∈ S → n ∈ S) ∧
  (∀ n ∈ S, (n + 5) ^ 2 ∈ S))
  (S : Set ℤ)
  (hS : IsLeast IsQualifying S)
: (Sᶜ ∩ {n | 0 < n} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q272
find_all (answer : Set ℤ) iff
  (IsQualifying : Set ℤ → Prop)
  (IsQualifying_def : ∀ S, IsQualifying S ↔
  (∀ n ∈ S, 0 < n) ∧
  2 ∈ S ∧
  (∀ n > 0, n ^ 2 ∈ S → n ∈ S) ∧
  (∀ n ∈ S, (n + 5) ^ 2 ∈ S))
  (S : Set ℤ)
  (hS : IsLeast IsQualifying S)
: (Sᶜ ∩ {n | 0 < n} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
