import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
In this problem, $a$ and $b$ are integers, such that $a \ge b.$

If $a+b\equiv 2\pmod{10}$ and $2a+b\equiv 1\pmod{10}$, then what is the last digit of $a-b$?

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q145.gt := fun (answer : ℤ) => answer = (6 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q145
find (answer : ℤ) s.t.
  (a : ℤ)
  (b : ℤ)
  (h₀ : b ≤ a)
  (h₁ : Int.emod (a + b) 10 = 2)
  (h₂ : Int.emod (2 * a + b) 10 = 1)
: (answer = Int.emod (a - b) 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q145
find_all (answer : ℤ) iff
  (a : ℤ)
  (b : ℤ)
  (h₀ : b ≤ a)
  (h₁ : Int.emod (a + b) 10 = 2)
  (h₂ : Int.emod (2 * a + b) 10 = 1)
: (answer = Int.emod (a - b) 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
