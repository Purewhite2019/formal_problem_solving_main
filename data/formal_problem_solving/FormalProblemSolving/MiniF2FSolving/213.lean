import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
For each plumbing repair job, Mr. Wrench charges $N$ dollars for coming out to the house plus $x$ dollars per hour that he works at the house. He charged $\$97$ for a one-hour repair job and $\$265$ for a five-hour repair job. What is his charge for a two-hour repair job?

## Answer
\$139
-/

-- # Formal Answer
noncomputable abbrev Q213.gt := fun (answer : ℝ) => answer = (139 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q213
find (answer : ℝ) s.t.
  (n : ℝ)
  (x : ℝ)
  (h₀ : n + x = 97)
  (h₁ : n + 5 * x = 265)
: (answer = n + 2 * x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q213
find_all (answer : ℝ) iff
  (n : ℝ)
  (x : ℝ)
  (h₀ : n + x = 97)
  (h₁ : n + 5 * x = 265)
: (answer = n + 2 * x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
