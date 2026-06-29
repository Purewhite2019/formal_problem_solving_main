import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The remainder can be defined for all real numbers $x$ and $y$ with $y \neq 0$ by $\text{rem} (x ,y)=x-y\left \lfloor \frac{x}{y} \right \rfloor$where $\left \lfloor \tfrac{x}{y} \right \rfloor$ denotes the greatest integer less than or equal to $\tfrac{x}{y}$. What is the value of $\text{rem} (\tfrac{3}{8}, -\tfrac{2}{5} )$?

$\textbf{(A) } -\frac{3}{8} \qquad \textbf{(B) } -\frac{1}{40} \qquad \textbf{(C) } 0 \qquad \textbf{(D) } \frac{3}{8} \qquad \textbf{(E) } \frac{31}{40}$

## Answer
(B) -1/40
-/

-- # Formal Answer
noncomputable abbrev Q183.gt := fun (answer : ℝ) => answer = (-(1 / 40 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q183
find (answer : ℝ) s.t.
  (f : ℝ → ℝ → ℝ)
  (h₀ : ∀ x y, y ≠ 0 → f x y = x - y * Int.floor (x / y))
: (answer = f (3 / 8) (-(2 / 5)))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q183
find_all (answer : ℝ) iff
  (f : ℝ → ℝ → ℝ)
  (h₀ : ∀ x y, y ≠ 0 → f x y = x - y * Int.floor (x / y))
: (answer = f (3 / 8) (-(2 / 5)))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
