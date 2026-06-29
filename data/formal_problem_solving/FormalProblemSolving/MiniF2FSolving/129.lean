import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose that $g(x)=f^{-1}(x)$. If $g(-15)=0$, $g(0)=3$, $g(3)=9$ and $g(9)=20$, what is $f(f(9))$?

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q129.gt := fun (answer : ℝ) => answer = (0 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q129
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_right_inv : Function.RightInverse g f)
  (h_left_inv : Function.LeftInverse g f)
  (h₀ : g (-15) = 0)
  (h₁ : g 0 = 3)
  (h₂ : g 3 = 9)
  (h₃ : g 9 = 20)
: (answer = f (f 9))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q129
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_right_inv : Function.RightInverse g f)
  (h_left_inv : Function.LeftInverse g f)
  (h₀ : g (-15) = 0)
  (h₁ : g 0 = 3)
  (h₂ : g 3 = 9)
  (h₃ : g 9 = 20)
: (answer = f (f 9))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
