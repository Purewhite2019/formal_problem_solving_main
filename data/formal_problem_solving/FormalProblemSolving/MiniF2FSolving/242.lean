import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The first and ninth terms of an arithmetic sequence are $\frac23$ and $\frac45$, respectively. What is the fifth term?

## Answer
$\frac{11}{15}$
-/

-- # Formal Answer
noncomputable abbrev Q242.gt := fun (answer : ℝ) => answer = (11 / 15 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q242
find (answer : ℝ) s.t.
  (a : ℕ → ℝ)
  (h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)
  (h₁ : a 1 = 2 / 3)
  (h₂ : a 9 = 4 / 5)
: (answer = a 5)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q242
find_all (answer : ℝ) iff
  (a : ℕ → ℝ)
  (h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)
  (h₁ : a 1 = 2 / 3)
  (h₂ : a 9 = 4 / 5)
: (answer = a 5)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
