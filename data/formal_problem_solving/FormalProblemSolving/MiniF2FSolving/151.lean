import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Suppose that $x,$ $y,$ and $z$ are three positive numbers that satisfy the equations $xyz = 1,$ $x + \frac {1}{z} = 5,$ and $y + \frac {1}{x} = 29.$ Then $z + \frac {1}{y} = \frac {m}{n},$ where $m$ and $n$ are [[relatively prime]] positive integers. Find $m + n$.


note: this is the type of problem that makes you think symmetry, but actually can be solved easily with substitution, and other normal technniques

## Answer
005
-/

-- # Formal Answer
noncomputable abbrev Q151.gt := fun (answer : ℕ) => answer = (5 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q151
find (answer : ℕ) s.t.
  (m : ℕ)
  (n : ℕ)
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h₀ : 0 < x ∧ 0 < y ∧ 0 < z)
  (h₁ : x * y * z = 1)
  (h₂ : x + 1 / z = 5)
  (h₃ : y + 1 / x = 29)
  (h₄ : z + 1 / y = ↑m / ↑n)
  (h₅ : Nat.Coprime m n)
: (m + n = answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q151
find_all (answer : ℕ) iff
  (m : ℕ)
  (n : ℕ)
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h₀ : 0 < x ∧ 0 < y ∧ 0 < z)
  (h₁ : x * y * z = 1)
  (h₂ : x + 1 / z = 5)
  (h₃ : y + 1 / x = 29)
  (h₄ : z + 1 / y = ↑m / ↑n)
  (h₅ : Nat.Coprime m n)
: (m + n = answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
