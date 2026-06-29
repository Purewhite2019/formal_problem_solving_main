import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given $m\geq 2$, denote by $b^{-1}$ the inverse of $b\pmod{m}$. That is, $b^{-1}$ is the residue for which $bb^{-1}\equiv 1\pmod{m}$. Sadie wonders if $(a+b)^{-1}$ is always congruent to $a^{-1}+b^{-1}$ (modulo $m$). She tries the example $a=2$, $b=3$, and $m=7$. Let $L$ be the residue of $(2+3)^{-1}\pmod{7}$, and let $R$ be the residue of $2^{-1}+3^{-1}\pmod{7}$, where $L$ and $R$ are integers from $0$ to $6$ (inclusive). Find $L-R$.

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q178.gt := fun (answer : ZMod 7) => answer = (1 : ZMod (7 : ℕ))

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q178
find (answer : ZMod 7) s.t.
  (l : ZMod 7)
  (r : ZMod 7)
  (h₀ : l = (2 + 3)⁻¹)
  (h₁ : r = 2⁻¹ + 3⁻¹)
: (answer = l - r)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q178
find_all (answer : ZMod 7) iff
  (l : ZMod 7)
  (r : ZMod 7)
  (h₀ : l = (2 + 3)⁻¹)
  (h₁ : r = 2⁻¹ + 3⁻¹)
: (answer = l - r)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
