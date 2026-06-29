import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $a$ and $b$ are real numbers, $a^2b^3=\frac{32}{27}$, and $\frac{a}{b^3}=\frac{27}{4}$, what is $a+b$?

## Answer
\frac83
-/

-- # Formal Answer
noncomputable abbrev Q337.gt := fun (answer : ℝ) => answer = (8 / 3 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q337
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (h₀ : a^2 * b^3 = 32 / 27)
  (h₁ : a / b^3 = 27 / 4)
: (answer = a + b)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q337
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (h₀ : a^2 * b^3 = 32 / 27)
  (h₁ : a / b^3 = 27 / 4)
: (answer = a + b)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
