import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The set of points $(x,y,z)$ that satisfy
\[2x = 3y = -z\]is a line.

The set of points $(x,y,z)$ that satisfy
\[6x = -y = -4z\]is another line.

Find the angle between these lines, in degrees.

## Answer
90^\circ
-/

-- # Formal Answer
noncomputable abbrev Q82.gt := fun (answer : ℝ) => answer = (90 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q82
find (answer : ℝ) s.t.
  (a : Fin 3 → ℝ)
  (b : Fin 3 → ℝ)
  (hab : a = ![3,2,-6] ∧ b = ![2,-12,-3])
: (answer = 180 * (Real.arccos ((∑ i, a i * b i) /
    (Real.sqrt ((∑ i, a i * a i) * (∑ i, b i * b i))))) / Real.pi)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q82
find_all (answer : ℝ) iff
  (a : Fin 3 → ℝ)
  (b : Fin 3 → ℝ)
  (hab : a = ![3,2,-6] ∧ b = ![2,-12,-3])
: (answer = 180 * (Real.arccos ((∑ i, a i * b i) /
    (Real.sqrt ((∑ i, a i * a i) * (∑ i, b i * b i))))) / Real.pi)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
