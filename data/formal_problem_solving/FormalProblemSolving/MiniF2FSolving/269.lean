import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Given that $p\ge 7$ is a prime number, evaluate $$1^{-1} \cdot 2^{-1} + 2^{-1} \cdot 3^{-1} + 3^{-1} \cdot 4^{-1} + \cdots + (p-2)^{-1} \cdot (p-1)^{-1} \pmod{p}.$$

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q269.gt := fun (p : ℕ) (answer : ZMod p) => answer = (2 : ZMod p)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q269
  (p : ℕ)
find (answer : ZMod p) s.t.
  (h₀ : Nat.Prime p)
  (h₁ : 7 ≤ p)
: (∑ k in Finset.Icc 1 (p-2), ((k : ZMod p)⁻¹ * ((k : ZMod p) + 1)⁻¹) = answer)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q269
  (p : ℕ)
find_all (answer : ZMod p) iff
  (h₀ : Nat.Prime p)
  (h₁ : 7 ≤ p)
: (∑ k in Finset.Icc 1 (p-2), ((k : ZMod p)⁻¹ * ((k : ZMod p) + 1)⁻¹) = answer)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
