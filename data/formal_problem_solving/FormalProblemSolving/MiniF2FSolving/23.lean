import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify $(9-4i)-(-3-4i)$.

## Answer
12
-/

-- # Formal Answer
noncomputable abbrev Q23.gt := fun (answer : ℂ) => answer = (12 : ℂ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q23
find (answer : ℂ) s.t.
  (q : ℂ)
  (e : ℂ)
  (h₀ : q = 9 - 4 * Complex.I)
  (h₁ : e = -3 - 4 * Complex.I)
: (answer = q - e)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q23
find_all (answer : ℂ) iff
  (q : ℂ)
  (e : ℂ)
  (h₀ : q = 9 - 4 * Complex.I)
  (h₁ : e = -3 - 4 * Complex.I)
: (answer = q - e)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
