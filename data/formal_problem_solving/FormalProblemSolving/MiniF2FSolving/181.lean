import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $x \equiv 4 \pmod{19}$ and $y \equiv 7 \pmod{19}$, then find the remainder when $(x + 1)^2 (y + 5)^3$ is divided by 19.

## Answer
13
-/

-- # Formal Answer
noncomputable abbrev Q181.gt := fun (answer : ℕ) => answer = (13 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q181
find (answer : ℕ) s.t.
  (x : ℕ)
  (y : ℕ)
  (h₀ : x % 19 = 4)
  (h₁ : y % 19 = 7)
: (answer = ((x + 1)^2 * (y + 5)^3) % 19)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q181
find_all (answer : ℕ) iff
  (x : ℕ)
  (y : ℕ)
  (h₀ : x % 19 = 4)
  (h₁ : y % 19 = 7)
: (answer = ((x + 1)^2 * (y + 5)^3) % 19)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
