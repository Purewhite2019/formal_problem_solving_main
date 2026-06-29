import FormalProblemSolving.Basic
open Real

/-
## Problem
The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?

$\textbf{(A) } -\sqrt{3}  \qquad\textbf{(B) } -\frac{2\sqrt{3}}{3} \qquad\textbf{(C) } -\frac{\sqrt{3}}{3} \qquad\textbf{(D) } \sqrt{3} \qquad\textbf{(E) } 3$

## Answer
(B) -\frac{2\sqrt{3}}{3}
-/

-- # Formal Answer
noncomputable abbrev Q80.gt := fun (answer : ℝ) => answer = (-(2 * Real.sqrt 3 / 3) : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q80
find (answer : ℝ) s.t.
: (answer ≤ 0) ∧ (∃ (r : ℝ), (answer * r = (2 : ℝ)) ∧ (answer * (r ^ (3 : ℕ) : ℝ) = (6 : ℝ)))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q80
find_all (answer : ℝ) iff
: (answer ≤ 0) ∧ (∃ (r : ℝ), (answer * r = (2 : ℝ)) ∧ (answer * (r ^ (3 : ℕ) : ℝ) = (6 : ℝ)))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
