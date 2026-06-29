import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Expand and simplify completely: \begin{align*}
x\left(x(1+x)+2x\right)-3(x^2-x+2)
\end{align*}

## Answer
x^3+3x-6
-/

-- # Formal Answer
noncomputable abbrev Q120.gt := fun (answer : ℝ → ℝ) => answer = fun (x : ℝ) => (((x ^ (3 : ℕ) : ℝ) + ((3 : ℝ) * x : ℝ) : ℝ) - (6 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q120
find (answer : ℝ → ℝ) s.t.
: (∀ x:ℝ, x * (x * (1 + x) + 2 * x) - 3 * (x^2 - x + 2) = answer x)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q120
find_all (answer : ℝ → ℝ) iff
: (∀ x:ℝ, x * (x * (1 + x) + 2 * x) - 3 * (x^2 - x + 2) = answer x)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
