import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The Fibonacci sequence is the sequence 1, 1, 2, 3, 5, $\ldots$ where each term is the sum of the previous two terms. What is the remainder when the $100^{\mathrm{th}}$ term of the sequence is divided by 4?

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q278.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q278
find (answer : ℕ) s.t.
: (answer = (Nat.fib 100) % 4)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q278
find_all (answer : ℕ) iff
: (answer = (Nat.fib 100) % 4)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
