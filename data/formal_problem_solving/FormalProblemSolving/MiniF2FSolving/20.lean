import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $f$ be a function satisfying $f(xy) = \frac{f(x)}y$ for all positive real numbers $x$ and $y$. If $f(500) =3$, what is the value of $f(600)$?

$(\mathrm{A})\ 1 \qquad (\mathrm{B})\ 2 \qquad (\mathrm{C})\ \frac52 \qquad (\mathrm{D})\ 3 \qquad (\mathrm{E})\ \frac{18}5$

## Answer
$\frac{5}{2}$
-/

-- # Formal Answer
noncomputable abbrev Q20.gt := fun (answer : ℝ) => answer = (5 / 2 : ℝ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q20
find (answer : ℝ) s.t.
  (f : ℝ → ℝ)
  (h₀ : ∀ x > 0, ∀ y > 0, f (x * y) = f x / y)
  (h₁ : f 500 = 3)
: (answer = f 600)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q20
find_all (answer : ℝ) iff
  (f : ℝ → ℝ)
  (h₀ : ∀ x > 0, ∀ y > 0, f (x * y) = f x / y)
  (h₁ : f 500 = 3)
: (answer = f 600)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
