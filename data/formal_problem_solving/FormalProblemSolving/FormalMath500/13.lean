import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Louis and Jack are sharing a pie.  Louis eats $\frac{1}{5}$ of the pie, and Jack eats $\frac{2}{3}$ of the pie.  How much of the pie do Louis and Jack eat altogether?

## Answer
\frac{13}{15}
-/

-- # Formal Answer
noncomputable abbrev Q13.gt := fun (answer : ℚ) => answer = (13 / 15 : ℚ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q13
find (answer : ℚ) s.t.
  (louis_eats : ℚ)
  (jack_eats : ℚ)
  (h₀ : louis_eats = 1/5)
  (h₁ : jack_eats = 2/3)
: (louis_eats + jack_eats = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q13
find_all (answer : ℚ) iff
  (louis_eats : ℚ)
  (jack_eats : ℚ)
  (h₀ : louis_eats = 1/5)
  (h₁ : jack_eats = 2/3)
: (louis_eats + jack_eats = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
