import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The least common multiple of two numbers is 3720, and their greatest common divisor is 8. Given that one of the numbers is 120, what is the other number?

## Answer
248
-/

-- # Formal Answer
noncomputable abbrev Q349.gt := fun (answer : ℕ) => answer = (248 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q349
find (answer : ℕ) s.t.
  (b : ℕ)
  (h₀ : Nat.lcm 120 b = 3720)
  (h₁ : Nat.gcd 120 b = 8)
: (answer = b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q349
find_all (answer : ℕ) iff
  (b : ℕ)
  (h₀ : Nat.lcm 120 b = 3720)
  (h₁ : Nat.gcd 120 b = 8)
: (answer = b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
