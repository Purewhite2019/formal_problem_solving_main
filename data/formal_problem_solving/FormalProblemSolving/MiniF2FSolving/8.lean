import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the average of the two smallest positive integer solutions to the congruence $$14u \equiv 46 \pmod{100}~?$$

## Answer
64
-/

-- # Formal Answer
noncomputable abbrev Q8.gt := fun (answer : ℚ) => answer = (64 : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q8
find (answer : ℚ) s.t.
  (u : ℕ)
  (v : ℕ)
  (S : Set ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ 0 < n ∧ (14 * n) % 100 = 46)
  (h₁ : IsLeast S u)
  (h₂ : IsLeast (S \ {u}) v)
: (answer = ((u + v) : ℚ) / 2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q8
find_all (answer : ℚ) iff
  (u : ℕ)
  (v : ℕ)
  (S : Set ℕ)
  (h₀ : ∀ (n : ℕ), n ∈ S ↔ 0 < n ∧ (14 * n) % 100 = 46)
  (h₁ : IsLeast S u)
  (h₂ : IsLeast (S \ {u}) v)
: (answer = ((u + v) : ℚ) / 2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
