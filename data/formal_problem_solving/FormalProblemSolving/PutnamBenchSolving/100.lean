import FormalProblemSolving.Basic
open Set Filter Topology Real Polynomial Function

/-
## Problem
Evaluate $\int_0^\infty t^{-1/2}e^{-1985(t+t^{-1})}\,dt$. You may assume that $\int_{-\infty}^\infty e^{-x^2}\,dx = \sqrt{\pi}$.

## Answer
$\sqrt{\pi / 1985} \cdot e^{-3970}$
-/

-- # Formal Answer
noncomputable abbrev Q100.gt := fun (answer : ℝ) => answer = ((√((π : ℝ) / (1985 : ℝ) : ℝ) : ℝ) * (rexp (-3970 : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q100
find (answer : ℝ) s.t.
  (fact : ∫ x in Set.univ, Real.exp (- x ^ 2) = Real.sqrt Real.pi)
: (∫ t in Set.Ioi 0, t ^ (- (1 : ℝ) / 2) * Real.exp (-1985 * (t + t ^ (-(1 : ℝ)))) = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q100
find_all (answer : ℝ) iff
  (fact : ∫ x in Set.univ, Real.exp (- x ^ 2) = Real.sqrt Real.pi)
: (∫ t in Set.Ioi 0, t ^ (- (1 : ℝ) / 2) * Real.exp (-1985 * (t + t ^ (-(1 : ℝ)))) = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
