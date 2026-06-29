import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Consider the function \[ f(x) =
\begin{cases}
ax^2 & \text{if } x \geq a,\\
ax +2a& \text{if } x <a,
\end{cases}
\]where $a$ is some number.

What is the largest value of $a$ such that the graph of $y=f(x)$ intersects every horizontal line at least once?

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q160.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q160
find (answer : ℝ) s.t.
  (S : Set ℝ)
  (hS : S = { a : ℝ | Function.Surjective (λ x : ℝ => if x ≥ a then a * x^2 else a * x + 2 * a) })
: (IsGreatest S answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q160
find_all (answer : ℝ) iff
  (S : Set ℝ)
  (hS : S = { a : ℝ | Function.Surjective (λ x : ℝ => if x ≥ a then a * x^2 else a * x + 2 * a) })
: (IsGreatest S answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
