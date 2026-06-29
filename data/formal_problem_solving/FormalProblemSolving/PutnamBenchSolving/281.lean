import FormalProblemSolving.Basic
open Topology Filter

/-
## Problem
Determine all possible values of the expression
\[
A^3+B^3+C^3-3ABC
\]
where $A, B$, and $C$ are nonnegative integers.

## Answer
the set of nonnegative integers not congruent to 3 or 6 modulo 9
-/

-- # Formal Answer
noncomputable abbrev Q281.gt := fun (answer : Set ℤ) => answer = ({n : ℤ | n ≥ (0 : ℤ) ∧ ¬n ≡ (3 : ℤ) [ZMOD (9 : ℤ)] ∧ ¬n ≡ (6 : ℤ) [ZMOD (9 : ℤ)]} : Set ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q281
find (answer : Set ℤ) s.t.
: ({n : ℤ | ∃ A B C : ℤ, A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^3 + B^3 + C^3 - 3*A*B*C = n} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q281
find_all (answer : Set ℤ) iff
: ({n : ℤ | ∃ A B C : ℤ, A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^3 + B^3 + C^3 - 3*A*B*C = n} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
