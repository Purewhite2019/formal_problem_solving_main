import FormalProblemSolving.Basic
open Nat

/-
## Problem
Prove or disprove that there exists a positive real number $\alpha$ such that $[\alpha_n] - n$ is even for all integers $n > 0$. (Here $[x]$ denotes the greatest integer less than or equal to $x$.)

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q81.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q81
find (answer : Prop) s.t.
: (answer ↔ (∃ α : ℝ, α > 0 ∧ ∀ n : ℕ, n > 0 → Even (⌊α ^ n⌋ - n)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q81
find_all (answer : Prop) iff
: (answer ↔ (∃ α : ℝ, α > 0 ∧ ∀ n : ℕ, n > 0 → Even (⌊α ^ n⌋ - n)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
