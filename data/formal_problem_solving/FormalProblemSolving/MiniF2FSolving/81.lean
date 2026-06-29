import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find $x^2+y^2_{}$ if $x_{}^{}$ and $y_{}^{}$ are positive integers such that
$\begin{align*}
xy+x+y&=71, \\
x^2y+xy^2&=880.
\end{align*}$

## Answer
146
-/

-- # Formal Answer
noncomputable abbrev Q81.gt := fun (answer : ℕ) => answer = (146 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q81
find (answer : ℕ) s.t.
  (x : ℕ)
  (y : ℕ)
  (h₀ : 0 < x ∧ 0 < y)
  (h₁ : x * y + (x + y) = 71)
  (h₂ : x^2 * y + x * y^2 = 880)
: (answer = x^2 + y^2)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q81
find_all (answer : ℕ) iff
  (x : ℕ)
  (y : ℕ)
  (h₀ : 0 < x ∧ 0 < y)
  (h₁ : x * y + (x + y) = 71)
  (h₂ : x^2 * y + x * y^2 = 880)
: (answer = x^2 + y^2)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
