import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Complex numbers are often used when dealing with alternating current (AC) circuits. In the equation $V = IZ$, $V$ is voltage, $I$ is current, and $Z$ is a value known as impedance. If $V = 1+i$ and $Z=2-i$, find $I$.

## Answer
\frac{1}{5} + \frac{3}{5}i
-/

-- # Formal Answer
noncomputable abbrev Q293.gt := fun (answer : ℂ) => answer = ((1 / 5 : ℂ) + ((3 / 5 : ℂ) * (Complex.I : ℂ) : ℂ) : ℂ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q293
find (answer : ℂ) s.t.
  (v : ℂ)
  (i : ℂ)
  (z : ℂ)
  (h₀ : v = i * z)
  (h₁ : v = 1 + Complex.I)
  (h₂ : z = 2 - Complex.I)
: (answer = i)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q293
find_all (answer : ℂ) iff
  (v : ℂ)
  (i : ℂ)
  (z : ℂ)
  (h₀ : v = i * z)
  (h₁ : v = 1 + Complex.I)
  (h₂ : z = 2 - Complex.I)
: (answer = i)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
