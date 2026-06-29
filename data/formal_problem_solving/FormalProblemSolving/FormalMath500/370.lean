import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute
\[\sum_{n = 1}^\infty \frac{F_{n + 1}}{F_n F_{n + 2}},\]where $F_n$ denotes the $n$th Fibonacci number, so $F_0 = 0$ and $F_1 = 1.$

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q370.gt := fun (answer : ℝ) => answer = (2 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q370
find (answer : ℝ) s.t.
: (answer = ∑' (n : ℕ), (Nat.fib (n + 2) : ℝ) / (Nat.fib (n + 1) * Nat.fib (n + 3)))
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q370
find_all (answer : ℝ) iff
: (answer = ∑' (n : ℕ), (Nat.fib (n + 2) : ℝ) / (Nat.fib (n + 1) * Nat.fib (n + 3)))
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
