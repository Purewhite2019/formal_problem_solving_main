import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify $\frac{1+\sqrt{2}}{2+\sqrt{3}}$. Your solution can be converted to the form $A(1+\sqrt{B})-(\sqrt{C}+\sqrt{D})$, where $A$, $B$, $C$, and $D$ are positive integers. What is $A+B+C+D$?

## Answer
13
-/

-- # Formal Answer
noncomputable abbrev Q194.gt := fun (answer : ℕ) => answer = (13 : ℕ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q194
find (answer : ℕ) s.t.
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (D : ℕ)
  (h₀ : A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0)
  (h₁ : (1 + √2) / (2 + √3) = A * (1 + √B) - (√C + √D))
: (A + B + C + D = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q194
find_all (answer : ℕ) iff
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (D : ℕ)
  (h₀ : A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0)
  (h₁ : (1 + √2) / (2 + √3) = A * (1 + √B) - (√C + √D))
: (A + B + C + D = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
