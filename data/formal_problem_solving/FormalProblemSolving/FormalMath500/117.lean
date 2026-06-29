import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Three pencils and a jumbo eraser cost $\$1.24$. Five pencils and a jumbo eraser cost $\$1.82$. No prices include tax. In cents, what is the cost of a pencil?

## Answer
29
-/

-- # Formal Answer
noncomputable abbrev Q117.gt := fun (answer : ℕ) => answer = (29 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q117
find (answer : ℕ) s.t.
  (pencil_cost : ℕ)
  (eraser_cost : ℕ)
  (h₀ : 3 * pencil_cost + eraser_cost = 124)
  (h₁ : 5 * pencil_cost + eraser_cost = 182)
: (pencil_cost = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q117
find_all (answer : ℕ) iff
  (pencil_cost : ℕ)
  (eraser_cost : ℕ)
  (h₀ : 3 * pencil_cost + eraser_cost = 124)
  (h₁ : 5 * pencil_cost + eraser_cost = 182)
: (pencil_cost = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
