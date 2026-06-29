import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
Find the positive integer $n\,$ for which
$
\lfloor\log_2{1}\rfloor+\lfloor\log_2{2}\rfloor+\lfloor\log_2{3}\rfloor+\cdots+\lfloor\log_2{n}\rfloor=1994
$
(For real $x\,$, $\lfloor x\rfloor\,$ is the greatest integer $\le x.\,$)

## Answer
312
-/

-- # Formal Answer
noncomputable abbrev Q44.gt := fun (answer : ℕ) => answer = (312 : ℕ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q44
find (answer : ℕ) s.t.
  (n : ℕ)
  (h₀ : 0 < n)
  (h₁ : ∑ k in Finset.Icc 1 n, Int.floor (Real.logb 2 k) = 1994)
: (answer = n)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q44
find_all (answer : ℕ) iff
  (n : ℕ)
  (h₀ : 0 < n)
  (h₁ : ∑ k in Finset.Icc 1 n, Int.floor (Real.logb 2 k) = 1994)
: (answer = n)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
