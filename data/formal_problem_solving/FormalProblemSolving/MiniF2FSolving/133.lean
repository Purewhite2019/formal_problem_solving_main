import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute $(3^{-1}+5^{-1})^{-1}\pmod{31}$. Express your answer as an integer from $0$ to $30$, inclusive.

## Answer
29
-/

-- # Formal Answer
noncomputable abbrev Q133.gt := fun (answer : ZMod 31) => answer = (29 : ZMod (31 : ℕ))

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q133
find (answer : ZMod 31) s.t.
  (x : ZMod 31)
  (y : ZMod 31)
  (z : ZMod 31)
  (h₀ : x = 3⁻¹)
  (h₁ : y = 5⁻¹)
  (h₂ : z = (x + y)⁻¹)
: (answer = z)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q133
find_all (answer : ZMod 31) iff
  (x : ZMod 31)
  (y : ZMod 31)
  (z : ZMod 31)
  (h₀ : x = 3⁻¹)
  (h₁ : y = 5⁻¹)
  (h₂ : z = (x + y)⁻¹)
: (answer = z)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
