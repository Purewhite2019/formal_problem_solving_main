import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $p$, $q$, and $r$ be constants. One solution to the equation $(x-p)(x-q) = (r-p)(r-q)$ is $x=r$. Find the other solution in terms of $p$, $q$, and $r$.

## Answer
p+q-r
-/

-- # Formal Answer
noncomputable abbrev Q31.gt := fun (p : ℂ) (q : ℂ) (r : ℂ) (answer : ℂ) => answer = ((p + q : ℂ) - r : ℂ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q31
  (p : ℂ)
  (q : ℂ)
  (r : ℂ)
find (answer : ℂ) s.t.
  (x : ℂ)
  (h₀ : (x - p) * (x - q) = (r - p) * (r - q))
  (h₁ : x ≠ r)
: (answer = x)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q31
  (p : ℂ)
  (q : ℂ)
  (r : ℂ)
find_all (answer : ℂ) iff
  (x : ℂ)
  (h₀ : (x - p) * (x - q) = (r - p) * (r - q))
  (h₁ : x ≠ r)
: (answer = x)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
