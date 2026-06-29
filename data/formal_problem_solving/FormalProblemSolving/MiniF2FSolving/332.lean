import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sum of the first 5 terms of an arithmetic series is $70$.  The sum of the first 10 terms of this  arithmetic series is $210$.  What is the first term of the series?

## Answer
\frac{42}{5}
-/

-- # Formal Answer
noncomputable abbrev Q332.gt := fun (answer : ℝ) => answer = (42 / 5 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q332
find (answer : ℝ) s.t.
  (a : ℝ)
  (d : ℝ)
  (h₀ : ∑ k in (Finset.range 5), (a + k * d) = 70)
  (h₁ : ∑ k in (Finset.range 10), (a + k * d) = 210)
: (answer = a)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q332
find_all (answer : ℝ) iff
  (a : ℝ)
  (d : ℝ)
  (h₀ : ∑ k in (Finset.range 5), (a + k * d) = 70)
  (h₁ : ∑ k in (Finset.range 10), (a + k * d) = 210)
: (answer = a)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
