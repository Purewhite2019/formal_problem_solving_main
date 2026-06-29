import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $a,b,$ and $c$ are positive [[real number]]s such that $a(b+c) = 152, b(c+a) = 162,$ and $c(a+b) = 170$, then $abc$ is

$\mathrm{(A)}\ 672
\qquad\mathrm{(B)}\ 688
\qquad\mathrm{(C)}\ 704
\qquad\mathrm{(D)}\ 720
\qquad\mathrm{(E)}\ 750$

## Answer
720
-/

-- # Formal Answer
noncomputable abbrev Q300.gt := fun (answer : ℝ) => answer = (720 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q300
find (answer : ℝ) s.t.
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : a * (b + c) = 152)
  (h₂ : b * (c + a) = 162)
  (h₃ : c * (a + b) = 170)
: (answer = a * b * c)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q300
find_all (answer : ℝ) iff
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : a * (b + c) = 152)
  (h₂ : b * (c + a) = 162)
  (h₃ : c * (a + b) = 170)
: (answer = a * b * c)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
