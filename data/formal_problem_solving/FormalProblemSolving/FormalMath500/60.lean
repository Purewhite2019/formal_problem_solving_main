import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
I have a deck of $54$ cards, and I deal all of the cards to $x$ players, with each player getting $y$ cards. If $x$ is at least $2$ and $y$ is at least $5$, then how many possible values of $x$ are there?

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q60.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q60
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (hS : S = {x:ℕ | ∃ y : ℕ, x*y = 54 ∧ x ≥ 2 ∧ y ≥ 5})
: (answer = S.encard)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q60
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (hS : S = {x:ℕ | ∃ y : ℕ, x*y = 54 ∧ x ≥ 2 ∧ y ≥ 5})
: (answer = S.encard)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
