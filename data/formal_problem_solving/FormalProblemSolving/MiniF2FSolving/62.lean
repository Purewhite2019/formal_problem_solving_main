import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $Q = 11-5i$, $E = 11+5i$, and $D = 2i$, find $Q\cdot E \cdot D$.

## Answer
292i
-/

-- # Formal Answer
noncomputable abbrev Q62.gt := fun (answer : ℂ) => answer = ((292 : ℂ) * (Complex.I : ℂ) : ℂ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q62
find (answer : ℂ) s.t.
  (q : ℂ)
  (e : ℂ)
  (d : ℂ)
  (h₀ : q = 11 - (5 * Complex.I))
  (h₁ : e = 11 + (5 * Complex.I))
  (h₂ : d = 2 * Complex.I)
: (answer = q * e * d)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q62
find_all (answer : ℂ) iff
  (q : ℂ)
  (e : ℂ)
  (d : ℂ)
  (h₀ : q = 11 - (5 * Complex.I))
  (h₁ : e = 11 + (5 * Complex.I))
  (h₂ : d = 2 * Complex.I)
: (answer = q * e * d)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
