import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
If $x$ and $y$ are positive integers for which $2^x3^y=1296$, what is the value of $x+y$?

$(\mathrm {A})\ 8 \qquad (\mathrm {B})\ 9 \qquad (\mathrm {C})\ 10 \qquad (\mathrm {D})\ 11 \qquad (\mathrm {E})\ 12$

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q73.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q73
find (answer : ℕ) s.t.
  (x : ℕ)
  (y : ℕ)
  (h₀ : 2^x * 3^y = 1296)
: (answer = x + y)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q73
find_all (answer : ℕ) iff
  (x : ℕ)
  (y : ℕ)
  (h₀ : 2^x * 3^y = 1296)
: (answer = x + y)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
