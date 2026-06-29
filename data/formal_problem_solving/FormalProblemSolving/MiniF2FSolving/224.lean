import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given a [[nonnegative]] real number $x$, let $\langle x\rangle$ denote the fractional part of $x$; that is, $\langle x\rangle=x-\lfloor x\rfloor$, where $\lfloor x\rfloor$ denotes the [[greatest integer]] less than or equal to $x$. Suppose that $a$ is positive, $\langle a^{-1}\rangle=\langle a^2\rangle$, and $2<a^2<3$. Find the value of $a^{12}-144a^{-1}$.

## Answer
233
-/

-- # Formal Answer
noncomputable abbrev Q224.gt := fun (answer : ℝ) => answer = (233 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q224
find (answer : ℝ) s.t.
  (a : ℝ)
  (h₀ : 0 < a)
  (h₁ : Int.fract (1 / a) = Int.fract (a^2))
  (h₂ : 2 < a^2)
  (h₃ : a^2 < 3)
: (answer = a^12 - 144 * (1 / a))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q224
find_all (answer : ℝ) iff
  (a : ℝ)
  (h₀ : 0 < a)
  (h₁ : Int.fract (1 / a) = Int.fract (a^2))
  (h₂ : 2 < a^2)
  (h₃ : a^2 < 3)
: (answer = a^12 - 144 * (1 / a))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
