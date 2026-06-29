import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $a \star b = \dfrac{\left(\dfrac{1}{b} - \dfrac{1}{a}\right)}{(a - b)}$, express $3 \star 11$ as a common fraction.

## Answer
$\frac{1}{33}$
-/

-- # Formal Answer
noncomputable abbrev Q363.gt := fun (answer : ℚ) => answer = (1 / 33 : ℚ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q363
find (answer : ℚ) s.t.
  (s : ℚ → ℚ → ℚ)
  (h₀ : ∀ x y, x ≠ 0 → y ≠ 0 → s x y = (1/y - 1/x) / (x - y))
: (answer = s 3 11)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q363
find_all (answer : ℚ) iff
  (s : ℚ → ℚ → ℚ)
  (h₀ : ∀ x y, x ≠ 0 → y ≠ 0 → s x y = (1/y - 1/x) / (x - y))
: (answer = s 3 11)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
