import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Let $S = 2010 + 2011 + \cdots + 4018$. Compute the residue of $S$, modulo 2009.

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q217.gt := fun (answer : ℕ) => answer = (0 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q217
find (answer : ℕ) s.t.
  (S : ℕ)
  (h₀ : S = ∑ k in Finset.Icc 2010 4018, k)
: (answer = S % 2009)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q217
find_all (answer : ℕ) iff
  (S : ℕ)
  (h₀ : S = ∑ k in Finset.Icc 2010 4018, k)
: (answer = S % 2009)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
