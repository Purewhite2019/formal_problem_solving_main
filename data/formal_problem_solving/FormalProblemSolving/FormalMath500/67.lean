import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the smallest positive real number $C$ for which
\[\left\| \begin{pmatrix} 2 & 3 \\ 0 & -2 \end{pmatrix} \bold{v} \right\| \le C \|\bold{v}\|\]for all two-dimensional vectors $\bold{v}.$

Note that for a two-dimensional vector $\mathbf{a},$ $\|\mathbf{a}\|$ is the magnitude of $\mathbf{a}.$

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q67.gt := fun (answer : ℝ) => answer = (4 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q67
find (answer : ℝ) s.t.
: (∀ (v : ℝ × ℝ),
    Real.sqrt ((2 * v.1 + 3 * v.2)^2 + (0 * v.1 + (-2) * v.2)^2) ≤ answer * Real.sqrt (v.1^2 + v.2^2)) ∧ (∀ (C' : ℝ), (∀ (v : ℝ × ℝ),
    Real.sqrt ((2 * v.1 + 3 * v.2)^2 + (0 * v.1 + (-2) * v.2)^2) ≤ C' * Real.sqrt (v.1^2 + v.2^2)) → answer ≤ C')
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q67
find_all (answer : ℝ) iff
: (∀ (v : ℝ × ℝ),
    Real.sqrt ((2 * v.1 + 3 * v.2)^2 + (0 * v.1 + (-2) * v.2)^2) ≤ answer * Real.sqrt (v.1^2 + v.2^2)) ∧ (∀ (C' : ℝ), (∀ (v : ℝ × ℝ),
    Real.sqrt ((2 * v.1 + 3 * v.2)^2 + (0 * v.1 + (-2) * v.2)^2) ≤ C' * Real.sqrt (v.1^2 + v.2^2)) → answer ≤ C')
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
