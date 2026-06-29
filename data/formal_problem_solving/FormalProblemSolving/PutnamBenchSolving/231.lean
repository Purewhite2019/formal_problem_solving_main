import FormalProblemSolving.Basic
open Filter Topology Set

/-
## Problem
Is there an infinite sequence of real numbers $a_1, a_2, a_3, \dots$ such that \[ a_1^m + a_2^m + a_3^m + \cdots = m \] for every positive integer $m$?

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q231.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q231
find (answer : Prop) s.t.
: (answer ↔ (∃ a : ℕ → ℝ, ∀ m : ℕ, m > 0 → ∑' i : ℕ, (a i)^m = m))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q231
find_all (answer : Prop) iff
: (answer ↔ (∃ a : ℕ → ℝ, ∀ m : ℕ, m > 0 → ∑' i : ℕ, (a i)^m = m))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
