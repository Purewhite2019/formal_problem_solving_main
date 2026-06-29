import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The function $f_{}^{}$ has the property that, for each real number $x,\,$
<center>$f(x)+f(x-1) = x^2.\,$</center>
If $f(19)=94,\,$ what is the remainder when $f(94)\,$ is divided by $1000$?

## Answer
561
-/

-- # Formal Answer
noncomputable abbrev Q238.gt := fun (answer : ℤ) => answer = (561 : ℤ)

-- # Formal Problem (FPS)
namespace MiniF2FSolving_FPS
open scoped FPS

problem Q238
find (answer : ℤ) s.t.
  (f : ℤ → ℤ)
  (h0 : ∀ x, f x + f (x - 1) = x^2)
  (h1 : f 19 = 94)
: (answer = f 94 % 1000)
:= solve
-- Placeholder
all_goals sorry

end MiniF2FSolving_FPS

-- # Formal Problem (D-FPS)
namespace MiniF2FSolving_DFPS
open scoped DFPS

problem Q238
find_all (answer : ℤ) iff
  (f : ℤ → ℤ)
  (h0 : ∀ x, f x + f (x - 1) = x^2)
  (h1 : f 19 = 94)
: (answer = f 94 % 1000)
:= solve
case Answer := sorry
all_goals sorry

end MiniF2FSolving_DFPS
