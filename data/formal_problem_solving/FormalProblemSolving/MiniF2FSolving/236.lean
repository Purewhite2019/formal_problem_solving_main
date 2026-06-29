import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f(x) = 12x+7$ and $g(x) = 5x+2$ whenever $x$ is a positive integer. Define $h(x)$ to be the greatest common divisor of $f(x)$ and $g(x)$. What is the sum of all possible values of $h(x)$?

## Answer
12
-/

-- # Formal Answer
noncomputable abbrev Q236.gt := fun (answer : ℕ) => answer = (12 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q236
find (answer : ℕ) s.t.
  (f : ℕ+ → ℕ)
  (g : ℕ+ → ℕ)
  (h : ℕ+ → ℕ)
  (h₀ : ∀ x, f x = 12 * x + 7)
  (h₁ : ∀ x, g x = 5 * x + 2)
  (h₂ : ∀ x, h x = Nat.gcd (f x) (g x))
  (h₃ : Fintype (Set.range h))
: (answer= ∑ k in (Set.range h).toFinset, k)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q236
find_all (answer : ℕ) iff
  (f : ℕ+ → ℕ)
  (g : ℕ+ → ℕ)
  (h : ℕ+ → ℕ)
  (h₀ : ∀ x, f x = 12 * x + 7)
  (h₁ : ∀ x, g x = 5 * x + 2)
  (h₂ : ∀ x, h x = Nat.gcd (f x) (g x))
  (h₃ : Fintype (Set.range h))
: (answer= ∑ k in (Set.range h).toFinset, k)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
