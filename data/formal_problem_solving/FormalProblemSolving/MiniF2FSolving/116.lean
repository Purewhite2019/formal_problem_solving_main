import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $k={2008}^{2}+{2}^{2008}$. What is the units digit of $k^2+2^k$?

$\mathrm{(A)}\ 0\qquad\mathrm{(B)}\ 2\qquad\mathrm{(C)}\ 4\qquad\mathrm{(D)}\ 6\qquad\mathrm{(E)}\ 8$

## Answer
D, which corresponds to 6
-/

-- # Formal Answer
noncomputable abbrev Q116.gt := fun (answer : ℕ) => answer = (6 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q116
find (answer : ℕ) s.t.
  (k : ℕ)
  (h₀ : k = 2008^2 + 2^2008)
: (answer = (k^2 + 2^k) % 10)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q116
find_all (answer : ℕ) iff
  (k : ℕ)
  (h₀ : k = 2008^2 + 2^2008)
: (answer = (k^2 + 2^k) % 10)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
