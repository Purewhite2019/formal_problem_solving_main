import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Seven cubes, whose volumes are $1$, $8$, $27$, $64$, $125$, $216$, and $343$ cubic units, are stacked vertically to form a tower in which the volumes of the cubes decrease from bottom to top. Except for the bottom cube, the bottom face of each cube lies completely on top of the cube below it. What is the total surface area of the tower (including the bottom) in square units?

$ \textbf{(A)}\ 644\qquad\textbf{(B)}\ 658\qquad\textbf{(C)}\ 664\qquad\textbf{(D)}\ 720\qquad\textbf{(E)}\ 749 $

## Answer
658
-/

-- # Formal Answer
noncomputable abbrev Q266.gt := fun (answer : ℕ) => answer = (658 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q266
find (answer : ℕ) s.t.
  (a : ℕ → ℕ)
  (h₀ : (a 0)^3 = 343)
  (h₁ : (a 1)^3 = 216)
  (h₂ : (a 2)^3 = 125)
  (h₃ : (a 3)^3 = 64)
  (h₄ : (a 4)^3 = 27)
  (h₅ : (a 5)^3 = 8)
  (h₆ : (a 6)^3 = 1)
: (answer = (∑ k in Finset.range 7, 6 * (a k)^2) - 2 * (∑ k in Finset.Ico 1 7, (a k)^2))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q266
find_all (answer : ℕ) iff
  (a : ℕ → ℕ)
  (h₀ : (a 0)^3 = 343)
  (h₁ : (a 1)^3 = 216)
  (h₂ : (a 2)^3 = 125)
  (h₃ : (a 3)^3 = 64)
  (h₄ : (a 4)^3 = 27)
  (h₅ : (a 5)^3 = 8)
  (h₆ : (a 6)^3 = 1)
: (answer = (∑ k in Finset.range 7, 6 * (a k)^2) - 2 * (∑ k in Finset.Ico 1 7, (a k)^2))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
