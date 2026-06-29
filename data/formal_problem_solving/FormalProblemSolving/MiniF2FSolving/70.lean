import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify $(2-2i)(5+5i)$, where $i^2 = -1.$

## Answer
20
-/

-- # Formal Answer
noncomputable abbrev Q70.gt := fun (answer : ℂ) => answer = (20 : ℂ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q70
find (answer : ℂ) s.t.
  (q : ℂ)
  (e : ℂ)
  (h₀ : q = 2 - 2 * Complex.I)
  (h₁ : e = 5 + 5 * Complex.I)
: (answer = q * e)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q70
find_all (answer : ℂ) iff
  (q : ℂ)
  (e : ℂ)
  (h₀ : q = 2 - 2 * Complex.I)
  (h₁ : e = 5 + 5 * Complex.I)
: (answer = q * e)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
