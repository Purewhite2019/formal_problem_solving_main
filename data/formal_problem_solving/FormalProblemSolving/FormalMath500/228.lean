import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The number $(\sqrt{2}+\sqrt{3})^3$ can be written in the form $a\sqrt{2} + b\sqrt{3} + c\sqrt{6}$, where $a$, $b$, and $c$ are integers.  What is $a+b+c$?

## Answer
20
-/

-- # Formal Answer
noncomputable abbrev Q228.gt := fun (answer : ℤ) => answer = (20 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q228
find (answer : ℤ) s.t.
  (a : ℤ)
  (b : ℤ)
  (c : ℤ)
  (h₀ : (Real.sqrt 2 + Real.sqrt 3)^3 = a * Real.sqrt 2 + b * Real.sqrt 3 + c * Real.sqrt 6)
: (a + b + c = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q228
find_all (answer : ℤ) iff
  (a : ℤ)
  (b : ℤ)
  (c : ℤ)
  (h₀ : (Real.sqrt 2 + Real.sqrt 3)^3 = a * Real.sqrt 2 + b * Real.sqrt 3 + c * Real.sqrt 6)
: (a + b + c = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
