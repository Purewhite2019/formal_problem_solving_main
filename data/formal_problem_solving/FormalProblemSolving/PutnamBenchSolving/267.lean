import FormalProblemSolving.Basic
open Polynomial Filter Topology Real Set Nat List

/-
## Problem
Let $x_0,x_1,x_2,\dots$ be the sequence such that $x_0=1$ and for $n \geq 0$,
\[
x_{n+1} = \ln(e^{x_n} - x_n)
\]
(as usual, the function $\ln$ is the natural logarithm). Show that the infinite series
\[
x_0 + x_1 + x_2 + \cdots
\]
converges and find its sum.

## Answer
exp 1 - 1
-/

-- # Formal Answer
noncomputable abbrev Q267.gt := fun (answer : ℝ) => answer = ((rexp (1 : ℝ) : ℝ) - (1 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q267
find (answer : ℝ) s.t.
  (x : ℕ → ℝ)
  (hx0 : x 0 = 1)
  (hxn : ∀ n : ℕ, x (n + 1) = log (exp (x n) - (x n)))
: (∑' n : ℕ, x n = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q267
find_all (answer : ℝ) iff
  (x : ℕ → ℝ)
  (hx0 : x 0 = 1)
  (hxn : ∀ n : ℕ, x (n + 1) = log (exp (x n) - (x n)))
: (∑' n : ℕ, x n = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
