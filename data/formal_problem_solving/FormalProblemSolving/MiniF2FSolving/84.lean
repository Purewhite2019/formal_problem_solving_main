import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose that the [[root]]s of $x^3+3x^2+4x-11=0$ are $a$, $b$, and $c$, and that the roots of $x^3+rx^2+sx+t=0$ are $a+b$, $b+c$, and $c+a$. Find $t$.

## Answer
23
-/

-- # Formal Answer
noncomputable abbrev Q84.gt := fun (answer : ℂ) => answer = (23 : ℂ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q84
find (answer : ℂ) s.t.
  (a : ℂ)
  (b : ℂ)
  (c : ℂ)
  (r : ℂ)
  (s : ℂ)
  (t : ℂ)
  (h₁ : a + b + c = -3)
  (h₂ : a * b + b * c + c * a = 4)
  (h₃ : a * b * c = 11)
  (h₄ : ∀ x, x^3 + r * x^2 + s * x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a)))
: (answer = t)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q84
find_all (answer : ℂ) iff
  (a : ℂ)
  (b : ℂ)
  (c : ℂ)
  (r : ℂ)
  (s : ℂ)
  (t : ℂ)
  (h₁ : a + b + c = -3)
  (h₂ : a * b + b * c + c * a = 4)
  (h₃ : a * b * c = 11)
  (h₄ : ∀ x, x^3 + r * x^2 + s * x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a)))
: (answer = t)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
