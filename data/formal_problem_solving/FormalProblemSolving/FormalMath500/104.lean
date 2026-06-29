import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the smallest positive integer $n$ such that all the roots of $z^4 + z^2 + 1 = 0$ are $n^{\text{th}}$ roots of unity?

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q104.gt := fun (answer : ℕ) => answer = (6 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q104
find (answer : ℕ) s.t.
: (answer > 0) ∧ (∀ z : ℂ, z^4 + z^2 + 1 = 0 → z^answer = 1) ∧ (∀ m : ℕ, 0 < m → m < answer → ∃ z : ℂ, z^4 + z^2 + 1 = 0 ∧ z^m ≠ 1)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q104
find_all (answer : ℕ) iff
: (answer > 0) ∧ (∀ z : ℂ, z^4 + z^2 + 1 = 0 → z^answer = 1) ∧ (∀ m : ℕ, 0 < m → m < answer → ∃ z : ℂ, z^4 + z^2 + 1 = 0 ∧ z^m ≠ 1)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
