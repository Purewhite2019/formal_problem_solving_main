import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the digit in the hundredths place of the decimal equivalent of $\frac{9}{160}$?

## Answer
5
-/

-- # Formal Answer
noncomputable abbrev Q180.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q180
find (answer : ℕ) s.t.
: (answer = (Int.floor ((9:ℝ) / 160 * 100)).toNat % 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q180
find_all (answer : ℕ) iff
: (answer = (Int.floor ((9:ℝ) / 160 * 100)).toNat % 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
