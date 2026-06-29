import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Positive integers $a$, $b$, and $2009$, with $a<b<2009$, form a geometric sequence with an integer ratio. What is $a$?

## Answer
41
-/

-- # Formal Answer
noncomputable abbrev Q219.gt := fun (answer : ℕ) => answer = (41 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q219
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (h₀ : 0 < a ∧ a < b ∧ b < 2009)
  (h₁ : ∃ (r : ℕ), b = a * r ∧ 2009 = b * r)
: (a = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q219
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (h₀ : 0 < a ∧ a < b ∧ b < 2009)
  (h₁ : ∃ (r : ℕ), b = a * r ∧ 2009 = b * r)
: (a = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
