import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the largest integer less than $2010$ that has a remainder of $5$ when divided by $7,$ a remainder of $10$ when divided by $11,$ and a remainder of $10$ when divided by $13$?

## Answer
1440
-/

-- # Formal Answer
noncomputable abbrev Q270.gt := fun (answer : ℕ) => answer = (1440 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q270
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {n:ℕ | n < 2010 ∧ n % 7 = 5 ∧ n % 11 = 10 ∧ n % 13 = 10})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q270
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {n:ℕ | n < 2010 ∧ n % 7 = 5 ∧ n % 11 = 10 ∧ n % 13 = 10})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
