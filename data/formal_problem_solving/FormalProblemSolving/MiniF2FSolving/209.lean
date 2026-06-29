import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $x$ be a positive number such that $2x^2 = 4x + 9.$ If $x$ can be written in simplified form as $\dfrac{a + \sqrt{b}}{c}$ such that $a,$ $b,$ and $c$ are positive integers, what is $a + b + c$?

## Answer
26
-/

-- # Formal Answer
noncomputable abbrev Q209.gt := fun (answer : ℕ) => answer = (26 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q209
find (answer : ℕ) s.t.
  (x : ℝ)
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (h_pos : 0 < x)
  (h_eq : 2 * x ^ 2 = 4 * x + 9)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_simplified : ∀ d, d > 1 → d ∣ a → d ∣ c → ¬ d ^ 2 ∣ b)
  (h_repr : x = (a + Real.sqrt b) / c)
: (answer = a + b + c)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q209
find_all (answer : ℕ) iff
  (x : ℝ)
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (h_pos : 0 < x)
  (h_eq : 2 * x ^ 2 = 4 * x + 9)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h_simplified : ∀ d, d > 1 → d ∣ a → d ∣ c → ¬ d ^ 2 ∣ b)
  (h_repr : x = (a + Real.sqrt b) / c)
: (answer = a + b + c)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
