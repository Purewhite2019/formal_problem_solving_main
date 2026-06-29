import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If we write $\sqrt{2}+\frac{1}{\sqrt{2}} + \sqrt{3} + \frac{1}{\sqrt{3}}$ in the form $\dfrac{a\sqrt{2} + b\sqrt{3}}{c}$ such that $a$, $b$, and $c$ are positive integers and $c$ is as small as possible, then what is $a+b+c$?

## Answer
23
-/

-- # Formal Answer
noncomputable abbrev Q165.gt := fun (answer : ℕ) => answer = (23 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q165
find (answer : ℕ) s.t.
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (h₀ : a > 0 ∧ b > 0 ∧ c > 0)
  (h₁ : √2 + 1 / √2 + √3 + 1 / √3 = (a * √2 + b * √3) / c)
  (h₂ : ∀ c' : ℕ, c' > 0 → c' < c → ¬∃ a' b' : ℕ, a' > 0 ∧ b' > 0 ∧ √2 + 1 / √2 + √3 + 1 / √3 = (a' * √2 + b' * √3) / c')
: (a + b + c = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q165
find_all (answer : ℕ) iff
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (h₀ : a > 0 ∧ b > 0 ∧ c > 0)
  (h₁ : √2 + 1 / √2 + √3 + 1 / √3 = (a * √2 + b * √3) / c)
  (h₂ : ∀ c' : ℕ, c' > 0 → c' < c → ¬∃ a' b' : ℕ, a' > 0 ∧ b' > 0 ∧ √2 + 1 / √2 + √3 + 1 / √3 = (a' * √2 + b' * √3) / c')
: (a + b + c = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
