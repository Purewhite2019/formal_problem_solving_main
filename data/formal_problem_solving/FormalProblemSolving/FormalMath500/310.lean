import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Steve says to Jon, "I am thinking of a polynomial whose roots are all positive integers. The polynomial has the form $P(x) = 2x^3-2ax^2+(a^2-81)x-c$ for some positive integers $a$ and $c$. Can you tell me the values of $a$ and $c$?"

After some calculations, Jon says, "There is more than one such polynomial."

Steve says, "You're right. Here is the value of $a$." He writes down a positive integer and asks, "Can you tell me the value of $c$?"

Jon says, "There are still two possible values of $c$."

Find the sum of the two possible values of $c$.

## Answer
440
-/

-- # Formal Answer
noncomputable abbrev Q310.gt := fun (answer : ℕ) => answer = (440 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q310
find (answer : ℕ) s.t.
  (a0 : ℕ)
  (c1 : ℕ)
  (c2 : ℕ)
  (ha0 : a0 > 0)
  (hc1 : c1 > 0)
  (hc2 : c2 > 0)
  (hne : c1 ≠ c2)
  (h1 : ∃ (r11 r12 r13 : ℕ), r11 > 0 ∧ r12 > 0 ∧ r13 > 0 ∧ a0 = r11 + r12 + r13 ∧ a0^2 = 2*(r11*r12 + r11*r13 + r12*r13) + 81 ∧ c1 = 2*r11*r12*r13)
  (h2 : ∃ (r21 r22 r23 : ℕ), r21 > 0 ∧ r22 > 0 ∧ r23 > 0 ∧ a0 = r21 + r22 + r23 ∧ a0^2 = 2*(r21*r22 + r21*r23 + r22*r23) + 81 ∧ c2 = 2*r21*r22*r23)
  (h3 : ∀ c : ℕ, c > 0 → (∃ (r1 r2 r3 : ℕ), r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ a0 = r1 + r2 + r3 ∧ a0^2 = 2*(r1*r2 + r1*r3 + r2*r3) + 81 ∧ c = 2*r1*r2*r3) → (c = c1 ∨ c = c2))
: (answer = c1 + c2)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q310
find_all (answer : ℕ) iff
  (a0 : ℕ)
  (c1 : ℕ)
  (c2 : ℕ)
  (ha0 : a0 > 0)
  (hc1 : c1 > 0)
  (hc2 : c2 > 0)
  (hne : c1 ≠ c2)
  (h1 : ∃ (r11 r12 r13 : ℕ), r11 > 0 ∧ r12 > 0 ∧ r13 > 0 ∧ a0 = r11 + r12 + r13 ∧ a0^2 = 2*(r11*r12 + r11*r13 + r12*r13) + 81 ∧ c1 = 2*r11*r12*r13)
  (h2 : ∃ (r21 r22 r23 : ℕ), r21 > 0 ∧ r22 > 0 ∧ r23 > 0 ∧ a0 = r21 + r22 + r23 ∧ a0^2 = 2*(r21*r22 + r21*r23 + r22*r23) + 81 ∧ c2 = 2*r21*r22*r23)
  (h3 : ∀ c : ℕ, c > 0 → (∃ (r1 r2 r3 : ℕ), r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ a0 = r1 + r2 + r3 ∧ a0^2 = 2*(r1*r2 + r1*r3 + r2*r3) + 81 ∧ c = 2*r1*r2*r3) → (c = c1 ∨ c = c2))
: (answer = c1 + c2)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
