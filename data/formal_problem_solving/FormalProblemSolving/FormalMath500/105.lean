import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the degree measure of the least positive angle $\theta$ for which
\[\tan \theta = \frac{\cos 5^\circ \cos 20^\circ + \cos 35^\circ \cos 50^\circ - \sin 5^\circ \sin 20^\circ - \sin 35^\circ \sin 50^\circ}{\sin 5^\circ \cos 20^\circ - \sin 35^\circ \cos 50^\circ + \cos 5^\circ \sin 20^\circ - \cos 35^\circ \sin 50^\circ}.\]

## Answer
120^\circ
-/

-- # Formal Answer
noncomputable abbrev Q105.gt := fun (answer : ℝ) => answer = (120 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q105
find (answer : ℝ) s.t.
  (h_denom : (Real.sin (5 * π / 180) * Real.cos (20 * π / 180) -
              Real.sin (35 * π / 180) * Real.cos (50 * π / 180) +
              Real.cos (5 * π / 180) * Real.sin (20 * π / 180) -
              Real.cos (35 * π / 180) * Real.sin (50 * π / 180)) ≠ 0)
: (0 < answer) ∧ (Real.tan (answer * π / 180) =
    (Real.cos (5 * π / 180) * Real.cos (20 * π / 180) +
     Real.cos (35 * π / 180) * Real.cos (50 * π / 180) -
     Real.sin (5 * π / 180) * Real.sin (20 * π / 180) -
     Real.sin (35 * π / 180) * Real.sin (50 * π / 180)) /
    (Real.sin (5 * π / 180) * Real.cos (20 * π / 180) -
     Real.sin (35 * π / 180) * Real.cos (50 * π / 180) +
     Real.cos (5 * π / 180) * Real.sin (20 * π / 180) -
     Real.cos (35 * π / 180) * Real.sin (50 * π / 180))) ∧ (∀ (d : ℝ), 0 < d →
      Real.tan (d * π / 180) =
        (Real.cos (5 * π / 180) * Real.cos (20 * π / 180) +
         Real.cos (35 * π / 180) * Real.cos (50 * π / 180) -
         Real.sin (5 * π / 180) * Real.sin (20 * π / 180) -
         Real.sin (35 * π / 180) * Real.sin (50 * π / 180)) /
        (Real.sin (5 * π / 180) * Real.cos (20 * π / 180) -
         Real.sin (35 * π / 180) * Real.cos (50 * π / 180) +
         Real.cos (5 * π / 180) * Real.sin (20 * π / 180) -
         Real.cos (35 * π / 180) * Real.sin (50 * π / 180)) →
      answer ≤ d)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q105
find_all (answer : ℝ) iff
  (h_denom : (Real.sin (5 * π / 180) * Real.cos (20 * π / 180) -
              Real.sin (35 * π / 180) * Real.cos (50 * π / 180) +
              Real.cos (5 * π / 180) * Real.sin (20 * π / 180) -
              Real.cos (35 * π / 180) * Real.sin (50 * π / 180)) ≠ 0)
: (0 < answer) ∧ (Real.tan (answer * π / 180) =
    (Real.cos (5 * π / 180) * Real.cos (20 * π / 180) +
     Real.cos (35 * π / 180) * Real.cos (50 * π / 180) -
     Real.sin (5 * π / 180) * Real.sin (20 * π / 180) -
     Real.sin (35 * π / 180) * Real.sin (50 * π / 180)) /
    (Real.sin (5 * π / 180) * Real.cos (20 * π / 180) -
     Real.sin (35 * π / 180) * Real.cos (50 * π / 180) +
     Real.cos (5 * π / 180) * Real.sin (20 * π / 180) -
     Real.cos (35 * π / 180) * Real.sin (50 * π / 180))) ∧ (∀ (d : ℝ), 0 < d →
      Real.tan (d * π / 180) =
        (Real.cos (5 * π / 180) * Real.cos (20 * π / 180) +
         Real.cos (35 * π / 180) * Real.cos (50 * π / 180) -
         Real.sin (5 * π / 180) * Real.sin (20 * π / 180) -
         Real.sin (35 * π / 180) * Real.sin (50 * π / 180)) /
        (Real.sin (5 * π / 180) * Real.cos (20 * π / 180) -
         Real.sin (35 * π / 180) * Real.cos (50 * π / 180) +
         Real.cos (5 * π / 180) * Real.sin (20 * π / 180) -
         Real.cos (35 * π / 180) * Real.sin (50 * π / 180)) →
      answer ≤ d)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
