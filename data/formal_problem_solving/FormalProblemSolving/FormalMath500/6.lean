import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Simplify $\frac{1}{5}\cdot \frac{8}{7}\div \frac{12}{20}$.

## Answer
\frac{8}{21}
-/

-- # Formal Answer
noncomputable abbrev Q6.gt := fun (answer : ℚ) => answer = (8 / 21 : ℚ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q6
find (answer : ℚ) s.t.
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)
  (d : ℚ)
  (e : ℚ)
  (f : ℚ)
  (h₀ : a = 1)
  (h₁ : b = 5)
  (h₂ : c = 8)
  (h₃ : d = 7)
  (h₄ : e = 12)
  (h₅ : f = 20)
: ((a / b) * (c / d) / (e / f) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q6
find_all (answer : ℚ) iff
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)
  (d : ℚ)
  (e : ℚ)
  (f : ℚ)
  (h₀ : a = 1)
  (h₁ : b = 5)
  (h₂ : c = 8)
  (h₃ : d = 7)
  (h₄ : e = 12)
  (h₅ : f = 20)
: ((a / b) * (c / d) / (e / f) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
