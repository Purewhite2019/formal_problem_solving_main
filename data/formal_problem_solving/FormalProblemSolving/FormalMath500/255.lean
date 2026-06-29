import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the smallest positive perfect cube that can be written as the sum of three consecutive integers?

## Answer
27
-/

-- # Formal Answer
noncomputable abbrev Q255.gt := fun (answer : ℕ) => answer = (27 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q255
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hn : S = {y : ℕ  | ∃ k : ℕ, k>0 ∧ k^3 ≡ 0 [MOD 3] ∧ y = k^3})
: (IsLeast S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q255
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hn : S = {y : ℕ  | ∃ k : ℕ, k>0 ∧ k^3 ≡ 0 [MOD 3] ∧ y = k^3})
: (IsLeast S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
