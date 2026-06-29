import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$

## Answer
\left( 3, \frac{\pi}{2} \right)
-/

-- # Formal Answer
noncomputable abbrev Q102.gt := fun (answer : ℝ × ℝ) => answer = (((3 : ℝ), ((π : ℝ) / (2 : ℝ) : ℝ)) : ℝ × ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q102
find (answer : ℝ × ℝ) s.t.
  (x : ℝ)
  (y : ℝ)
  (r : ℝ)
  (θ : ℝ)
  (h₀ : x = 0)
  (h₁ : y = 3)
  (h₂ : r = Real.sqrt (x^2 + y^2))
  (h₃ : θ = if x = 0 ∧ y > 0 then Real.pi / 2 else 0)
: ((r, θ) = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q102
find_all (answer : ℝ × ℝ) iff
  (x : ℝ)
  (y : ℝ)
  (r : ℝ)
  (θ : ℝ)
  (h₀ : x = 0)
  (h₁ : y = 3)
  (h₂ : r = Real.sqrt (x^2 + y^2))
  (h₃ : θ = if x = 0 ∧ y > 0 then Real.pi / 2 else 0)
: ((r, θ) = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
