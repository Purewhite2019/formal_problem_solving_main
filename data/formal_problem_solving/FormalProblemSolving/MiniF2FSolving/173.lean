import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Compute the sum of all the roots of
$(2x+3)(x-4)+(2x+3)(x-6)=0 $

$ \textbf{(A) } \frac{7}{2}\qquad \textbf{(B) } 4\qquad \textbf{(C) } 5\qquad \textbf{(D) } 7\qquad \textbf{(E) } 13 $

## Answer
7 / 2
-/

-- # Formal Answer
noncomputable abbrev Q173.gt := fun (answer : ℝ) => answer = (7 / 2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q173
find (answer : ℝ) s.t.
  (f : ℂ → ℂ)
  (h₀ : ∀ x, f x = (2 * x + 3) * (x - 4) + (2 * x + 3) * (x - 6))
  (h₁ : Fintype (f ⁻¹' {0}))
: (answer = ∑ y in (f⁻¹' {0}).toFinset, y)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q173
find_all (answer : ℝ) iff
  (f : ℂ → ℂ)
  (h₀ : ∀ x, f x = (2 * x + 3) * (x - 4) + (2 * x + 3) * (x - 6))
  (h₁ : Fintype (f ⁻¹' {0}))
: (answer = ∑ y in (f⁻¹' {0}).toFinset, y)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
