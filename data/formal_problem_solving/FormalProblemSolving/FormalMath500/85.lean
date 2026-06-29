import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the dot product of $\begin{pmatrix} -2 \\ 0 \\ 7 \end{pmatrix}$ and $\begin{pmatrix} 3 \\ 4 \\ -5 \end{pmatrix}.$

## Answer
-41
-/

-- # Formal Answer
noncomputable abbrev Q85.gt := fun (answer : ℝ) => answer = (-41 : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q85
find (answer : ℝ) s.t.
  (v₁ : Fin 3 → ℝ)
  (v₂ : Fin 3 → ℝ)
  (h₁ : v₁ = ![ -2, 0, 7 ])
  (h₂ : v₂ = ![ 3, 4, -5 ])
: (dotProduct v₁ v₂ = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q85
find_all (answer : ℝ) iff
  (v₁ : Fin 3 → ℝ)
  (v₂ : Fin 3 → ℝ)
  (h₁ : v₁ = ![ -2, 0, 7 ])
  (h₂ : v₂ = ![ 3, 4, -5 ])
: (dotProduct v₁ v₂ = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
