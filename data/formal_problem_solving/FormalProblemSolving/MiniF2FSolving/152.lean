import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose that $|x_i| < 1$ for $i = 1, 2, \dots, n$.  Suppose further that
$
|x_1| + |x_2| + \dots + |x_n| = 19 + |x_1 + x_2 + \dots + x_n|.
$
What is the smallest possible value of $n$?

## Answer
20
-/

-- # Formal Answer
noncomputable abbrev Q152.gt := fun (answer : ℕ) => answer = (20 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q152
find (answer : ℕ) s.t.
: (IsLeast {n : ℕ | ∃ (x : ℕ → ℝ), (∀ i, i < n → |x i| < 1) ∧ (∑ i in Finset.range n, |x i|) = 19 + |∑ i in Finset.range n, x i|} answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q152
find_all (answer : ℕ) iff
: (IsLeast {n : ℕ | ∃ (x : ℕ → ℝ), (∀ i, i < n → |x i| < 1) ∧ (∑ i in Finset.range n, |x i|) = 19 + |∑ i in Finset.range n, x i|} answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
