import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let \[f(n) =
\begin{cases}
4n+3 &\text{if }n<a, \\
7n-12 &\text{if }n\ge{a}.
\end{cases}
\]Find $a$ if the graph of $y=f(n)$ is continuous.

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q305.gt := fun (answer : ℝ) => answer = (5 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q305
find (answer : ℝ) s.t.
  (a : ℝ)
  (f : ℝ → ℝ)
  (h_left : ∀ n, n < a → f n = 4 * n + 3)
  (h_right : ∀ n, a ≤ n → f n = 7 * n - 12)
  (h_cont : Continuous f)
: (answer = a)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q305
find_all (answer : ℝ) iff
  (a : ℝ)
  (f : ℝ → ℝ)
  (h_left : ∀ n, n < a → f n = 4 * n + 3)
  (h_right : ∀ n, a ≤ n → f n = 7 * n - 12)
  (h_cont : Continuous f)
: (answer = a)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
