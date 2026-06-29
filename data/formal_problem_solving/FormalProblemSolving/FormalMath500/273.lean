import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If Rosa's age is divided by 2, 3, 4, or 6, the remainder is 1. If her age is divided by 7, the remainder is 0. She is less than 75 years old. How many years old is Rosa?

## Answer
49
-/

-- # Formal Answer
noncomputable abbrev Q273.gt := fun (answer : ℕ) => answer = (49 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q273
find (answer : ℕ) s.t.
  (n : ℕ)
  (hn : n > 0 ∧ n % 2 = 1 ∧ n % 3 = 1 ∧ n % 4 = 1 ∧ n % 6 = 1 ∧ n % 7 = 0)
  (hn2 : n < 75)
: (answer = n)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q273
find_all (answer : ℕ) iff
  (n : ℕ)
  (hn : n > 0 ∧ n % 2 = 1 ∧ n % 3 = 1 ∧ n % 4 = 1 ∧ n % 6 = 1 ∧ n % 7 = 0)
  (hn2 : n < 75)
: (answer = n)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
