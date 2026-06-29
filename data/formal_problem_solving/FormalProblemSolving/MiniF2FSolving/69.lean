import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
What is the sum of the smallest and second-smallest positive integers $a$ satisfying the congruence $$27a\equiv 17 \pmod{40}~?$$

## Answer
62
-/

-- # Formal Answer
noncomputable abbrev Q69.gt := fun (answer : ℕ) => answer = (62 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q69
find (answer : ℕ) s.t.
  (S : Set ℕ)
  (u : ℕ)
  (v : ℕ)
  (h₀ : ∀ (a : ℕ), a ∈ S ↔ 0 < a ∧ 27 * a % 40 = 17)
  (h₁ : IsLeast S u)
  (h₂ : IsLeast (S \ {u}) v)
: (answer = u + v)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q69
find_all (answer : ℕ) iff
  (S : Set ℕ)
  (u : ℕ)
  (v : ℕ)
  (h₀ : ∀ (a : ℕ), a ∈ S ↔ 0 < a ∧ 27 * a % 40 = 17)
  (h₁ : IsLeast S u)
  (h₂ : IsLeast (S \ {u}) v)
: (answer = u + v)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
