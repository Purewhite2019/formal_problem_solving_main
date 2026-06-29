import FormalProblemSolving.Basic

/-
## Problem
Find all integer solutions $(p, r, q, s)$ of the equation $|p^r - q^s| = 1$, where $p$ and $q$ are prime and $r$ and $s$ are greater than $1$.

## Answer
{(3, 2, 2, 3), (2, 3, 3, 2)}
-/

-- # Formal Answer
noncomputable abbrev Q40.gt := fun (answer : Set (ℕ × ℕ × ℕ × ℕ)) => answer =
  (Insert.insert (((3 : ℕ), (((2 : ℕ), (((2 : ℕ), (3 : ℕ)) : ℕ × ℕ)) : ℕ × ℕ × ℕ)) : ℕ × ℕ × ℕ × ℕ)
      ({(((2 : ℕ), (((3 : ℕ), (((3 : ℕ), (2 : ℕ)) : ℕ × ℕ)) : ℕ × ℕ × ℕ)) : ℕ × ℕ × ℕ × ℕ)} : Set (ℕ × ℕ × ℕ × ℕ)) :
    Set (ℕ × ℕ × ℕ × ℕ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q40
find (answer : Set (ℕ × ℕ × ℕ × ℕ)) s.t.
: ({a : ℕ × ℕ × ℕ × ℕ | Nat.Prime a.1 ∧ Nat.Prime a.2.2.1 ∧ a.2.1 > 1 ∧ a.2.2.2 > 1 ∧ |(a.1^a.2.1 : ℤ) - a.2.2.1^a.2.2.2| = 1} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q40
find_all (answer : Set (ℕ × ℕ × ℕ × ℕ)) iff
: ({a : ℕ × ℕ × ℕ × ℕ | Nat.Prime a.1 ∧ Nat.Prime a.2.2.1 ∧ a.2.1 > 1 ∧ a.2.2.2 > 1 ∧ |(a.1^a.2.1 : ℤ) - a.2.2.1^a.2.2.2| = 1} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
