import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
In the complex plane, let $A$ be the set of solutions to $z^{3}-8=0$ and let $B$ be the set of solutions to $z^{3}-8z^{2}-8z+64=0.$ What is the greatest distance between a point of $A$ and a point of $B?$

$\textbf{(A) } 2\sqrt{3} \qquad \textbf{(B) } 6 \qquad \textbf{(C) } 9 \qquad \textbf{(D) } 2\sqrt{21} \qquad \textbf{(E) } 9+\sqrt{3}$

## Answer
$2\sqrt{21}$
-/

-- # Formal Answer
noncomputable abbrev Q313.gt := fun (answer : ℝ) => answer = ((2 : ℝ) * (√(21 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q313
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = {z : ℝ|∃ a b :ℂ, a^3 - 8 = 0 ∧ b^3 - 8 * b^2 - 8 * b + 64 = 0 ∧ z = Complex.abs (a - b)})
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q313
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = {z : ℝ|∃ a b :ℂ, a^3 - 8 = 0 ∧ b^3 - 8 * b^2 - 8 * b + 64 = 0 ∧ z = Complex.abs (a - b)})
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
