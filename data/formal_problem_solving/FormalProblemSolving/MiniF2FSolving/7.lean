import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For what real value of $k$ is $\frac{13-\sqrt{131}}{4}$ a root of $2x^2-13x+k$?

## Answer
$\frac{19}{4}$
-/

-- # Formal Answer
noncomputable abbrev Q7.gt := fun (answer : ℝ) => answer = (19 / 4 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q7
find (answer : ℝ) s.t.
  (k : ℝ)
  (x : ℝ)
  (h₀ : x = (13 - Real.sqrt 131) / 4)
  (h₁ : 2 * x^2 - 13 * x + k = 0)
: (answer = k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q7
find_all (answer : ℝ) iff
  (k : ℝ)
  (x : ℝ)
  (h₀ : x = (13 - Real.sqrt 131) / 4)
  (h₁ : 2 * x^2 - 13 * x + k = 0)
: (answer = k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
