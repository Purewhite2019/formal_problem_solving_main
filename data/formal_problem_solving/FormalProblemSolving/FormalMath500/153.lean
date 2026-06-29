import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the integer value of $x$ in the arithmetic sequence $3^2, x, 3^4$?

## Answer
45
-/

-- # Formal Answer
noncomputable abbrev Q153.gt := fun (answer : ℤ) => answer = (45 : ℤ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q153
find (answer : ℤ) s.t.
  (x : ℤ)
  (h₀ : 3^2 = 9)
  (h₁ : 3^4 = 81)
  (h₂ : ∃ d : ℤ, x = 9 + d ∧ 81 = x + d)
: (x = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q153
find_all (answer : ℤ) iff
  (x : ℤ)
  (h₀ : 3^2 = 9)
  (h₁ : 3^4 = 81)
  (h₂ : ∃ d : ℤ, x = 9 + d ∧ 81 = x + d)
: (x = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
