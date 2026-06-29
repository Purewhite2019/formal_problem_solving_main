import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $a\star b = 9a+2b-ab+5$, what is the value of $5\star1$?

## Answer
47
-/

-- # Formal Answer
noncomputable abbrev Q218.gt := fun (answer : ℝ) => answer = (47 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q218
find (answer : ℝ) s.t.
  (f : ℝ → ℝ → ℝ)
  (hf : ∀ a b : ℝ, f a b = 9*a+2*b-a*b+5)
: (f 5 1 = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q218
find_all (answer : ℝ) iff
  (f : ℝ → ℝ → ℝ)
  (hf : ∀ a b : ℝ, f a b = 9*a+2*b-a*b+5)
: (f 5 1 = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
