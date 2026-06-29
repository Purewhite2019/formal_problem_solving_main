import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the number of quadratic equations of the form $x^2 + ax + b = 0,$ such that whenever $c$ is a root of the equation, $c^2 - 2$ is also a root of the equation.

## Answer
6
-/

-- # Formal Answer
noncomputable abbrev Q337.gt := fun (answer : ℕ) => answer = (6 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q337
find (answer : ℕ) s.t.
  (S : Set (ℝ × ℝ))
  (hS : S = { ab : ℝ × ℝ | ∀ (c : ℂ), (c ^ 2 + (ab.1 : ℂ) * c + (ab.2 : ℂ) = 0) →
    ((c ^ 2 - 2) ^ 2 + (ab.1 : ℂ) * (c ^ 2 - 2) + (ab.2 : ℂ) = 0) })
  (h_fin : Set.Finite S)
: (answer = Nat.card S)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q337
find_all (answer : ℕ) iff
  (S : Set (ℝ × ℝ))
  (hS : S = { ab : ℝ × ℝ | ∀ (c : ℂ), (c ^ 2 + (ab.1 : ℂ) * c + (ab.2 : ℂ) = 0) →
    ((c ^ 2 - 2) ^ 2 + (ab.1 : ℂ) * (c ^ 2 - 2) + (ab.2 : ℂ) = 0) })
  (h_fin : Set.Finite S)
: (answer = Nat.card S)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
