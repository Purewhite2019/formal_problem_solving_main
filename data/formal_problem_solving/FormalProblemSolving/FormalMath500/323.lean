import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The parabola with equation $y=ax^2+bx+c$ and vertex $(h,k)$ is reflected about the line $y=k$. This results in the parabola with equation $y=dx^2+ex+f$.  Express $a+b+c+d+e+f$ in terms of $k.$

## Answer
2k
-/

-- # Formal Answer
noncomputable abbrev Q323.gt := fun (k : ℝ) (answer : ℝ) => answer = ((2 : ℝ) * k : ℝ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q323
  (k : ℝ)
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (e : ℝ)
  (f : ℝ)
  (h : ℝ)
  (ha : a ≠ 0)
  (hd : d ≠ 0)
  (h_vertex_original : h = -b / (2 * a) ∧ k = a * h ^ 2 + b * h + c)
  (h_reflection_eq : ∀ x : ℝ, d * x ^ 2 + e * x + f = 2 * k - (a * x ^ 2 + b * x + c))
  (h_vertex_reflected : h = -e / (2 * d) ∧ k = d * h ^ 2 + e * h + f)
: (a + b + c + d + e + f = answer)
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q323
  (k : ℝ)
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
  (e : ℝ)
  (f : ℝ)
  (h : ℝ)
  (ha : a ≠ 0)
  (hd : d ≠ 0)
  (h_vertex_original : h = -b / (2 * a) ∧ k = a * h ^ 2 + b * h + c)
  (h_reflection_eq : ∀ x : ℝ, d * x ^ 2 + e * x + f = 2 * k - (a * x ^ 2 + b * x + c))
  (h_vertex_reflected : h = -e / (2 * d) ∧ k = d * h ^ 2 + e * h + f)
: (a + b + c + d + e + f = answer)
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
