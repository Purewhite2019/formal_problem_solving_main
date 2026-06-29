import FormalProblemSolving.Basic

/-
## Problem
Find polynomials $f(x)$,$g(x)$, and $h(x)$, if they exist, such that for all $x$, \[|f(x)|-|g(x)|+h(x) = \begin{cases} -1 & \mbox{if $x<-1$} \\3x+2 & \mbox{if $-1 \leq x \leq 0$} \\-2x+2 & \mbox{if $x>0$.}\end{cases}\]?

## Answer
True
-/

-- # Formal Answer
noncomputable abbrev Q179.gt := fun (answer : Prop) => answer = True

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q179
find (answer : Prop) s.t.
: (answer ↔ ∃ f g h : Polynomial ℝ, ∀ x : ℝ, |f.eval x| - |g.eval x| + h.eval x = if x < -1 then -1 else (if (x ≤ 0) then 3 * x + 2 else -2 * x + 2))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q179
find_all (answer : Prop) iff
: (answer ↔ ∃ f g h : Polynomial ℝ, ∀ x : ℝ, |f.eval x| - |g.eval x| + h.eval x = if x < -1 then -1 else (if (x ≤ 0) then 3 * x + 2 else -2 * x + 2))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
