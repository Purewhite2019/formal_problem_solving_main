import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $F_1 = (10,2)$ and $F_ 2= (-16,2).$  Then the set of points $P$ such that
\[|PF_1 - PF_2| = 24\]form a hyperbola.  The equation of this hyperbola can be written as
\[\frac{(x - h)^2}{a^2} - \frac{(y - k)^2}{b^2} = 1.\]Find $h + k + a + b.$

## Answer
16
-/

-- # Formal Answer
noncomputable abbrev Q360.gt := fun (answer : ℝ) => answer = (16 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q360
find (answer : ℝ) s.t.
  (h : ℝ)
  (k : ℝ)
  (a : ℝ)
  (b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (h_iff : ∀ (P : ℝ × ℝ),
      |Real.sqrt ((P.1 - 10)^2 + (P.2 - 2)^2) - Real.sqrt ((P.1 + 16)^2 + (P.2 - 2)^2)| = 24
      ↔ (P.1 - h)^2 / a^2 - (P.2 - k)^2 / b^2 = 1)
: (h + k + a + b = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q360
find_all (answer : ℝ) iff
  (h : ℝ)
  (k : ℝ)
  (a : ℝ)
  (b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (h_iff : ∀ (P : ℝ × ℝ),
      |Real.sqrt ((P.1 - 10)^2 + (P.2 - 2)^2) - Real.sqrt ((P.1 + 16)^2 + (P.2 - 2)^2)| = 24
      ↔ (P.1 - h)^2 / a^2 - (P.2 - k)^2 / b^2 = 1)
: (h + k + a + b = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
