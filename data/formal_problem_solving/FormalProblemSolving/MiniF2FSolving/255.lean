import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The sum of the squares of three consecutive positive even numbers is $12296$. Find the product of the three numbers divided by $8$.

## Answer
32736
-/

-- # Formal Answer
noncomputable abbrev Q255.gt := fun (answer : ℕ) => answer = (32736 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q255
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : Even n)
  (h₁ : ((n:ℤ) - 2)^2 + (n:ℤ)^2 + ((n:ℤ) + 2)^2 = 12296)
: (answer = ((n - 2) * n * (n + 2)) / 8)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q255
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : Even n)
  (h₁ : ((n:ℤ) - 2)^2 + (n:ℤ)^2 + ((n:ℤ) + 2)^2 = 12296)
: (answer = ((n - 2) * n * (n + 2)) / 8)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
