import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $x=\frac{\sum\limits_{n=1}^{44} \cos n^\circ}{\sum\limits_{n=1}^{44} \sin n^\circ}$. What is the greatest integer that does not exceed $100x$?

## Answer
241
-/

-- # Formal Answer
noncomputable abbrev Q27.gt := fun (answer : ℤ) => answer = (241 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q27
find (answer : ℤ) s.t.
  (x : ℝ)
  (h₀ : x = (∑ n in Finset.Icc (1 : ℕ) 44, Real.cos (n * π / 180)) / (∑ n in Finset.Icc (1 : ℕ) 44, Real.sin (n * π / 180)))
: (answer = Int.floor (100 * x))
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q27
find_all (answer : ℤ) iff
  (x : ℝ)
  (h₀ : x = (∑ n in Finset.Icc (1 : ℕ) 44, Real.cos (n * π / 180)) / (∑ n in Finset.Icc (1 : ℕ) 44, Real.sin (n * π / 180)))
: (answer = Int.floor (100 * x))
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
