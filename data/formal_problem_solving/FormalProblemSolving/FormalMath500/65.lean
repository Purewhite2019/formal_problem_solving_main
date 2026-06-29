import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Six witches and ten sorcerers are at an arcane mixer. The witches have decided to shake hands with each of the sorcerers. None of the witches bothered to shake each other's hands, since they are all good friends already, and the sorcerers all sort of hate each other and did not shake hands with other sorcerers. How many handshakes took place at the mixer?

## Answer
60
-/

-- # Formal Answer
noncomputable abbrev Q65.gt := fun (answer : ℕ) => answer = (60 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q65
find (answer : ℕ) s.t.
  (witches : ℕ)
  (sorcerers : ℕ)
  (h₀ : witches = 6)
  (h₁ : sorcerers = 10)
: (witches * sorcerers = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q65
find_all (answer : ℕ) iff
  (witches : ℕ)
  (sorcerers : ℕ)
  (h₀ : witches = 6)
  (h₁ : sorcerers = 10)
: (witches * sorcerers = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
