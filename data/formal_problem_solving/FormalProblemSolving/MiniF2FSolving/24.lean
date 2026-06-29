import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f$ be a [[function]] for which $f\left(\dfrac{x}{3}\right) = x^2 + x + 1$. Find the sum of all values of $z$ for which $f(3z) = 7$.

$\text {(A)}\ -1/3 \qquad \text {(B)}\ -1/9 \qquad \text {(C)}\ 0 \qquad \text {(D)}\ 5/9 \qquad \text {(E)}\ 5/3$

## Answer
- 1 / 9
-/

-- # Formal Answer
noncomputable abbrev Q24.gt := fun (answer : ℂ) => answer = (-1 / 9 : ℂ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q24
find (answer : ℂ) s.t.
  (f : ℂ → ℂ)
  (h₀ : ∀ x, f (x / 3) = x^2 + x + 1)
  (h₁ : Fintype (f ⁻¹' {7}))
: (∑ y in (f⁻¹' {7}).toFinset, y / 3 = answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q24
find_all (answer : ℂ) iff
  (f : ℂ → ℂ)
  (h₀ : ∀ x, f (x / 3) = x^2 + x + 1)
  (h₁ : Fintype (f ⁻¹' {7}))
: (∑ y in (f⁻¹' {7}).toFinset, y / 3 = answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
