import FormalProblemSolving.Basic
open MvPolynomial Real Nat

/-
## Problem
Evaluate
\[
\int_2^4 \frac{\sqrt{\ln(9-x)}\,dx}{\sqrt{\ln(9-x)}+\sqrt{\ln(x+3)}}.
\]

## Answer
1
-/

-- # Formal Answer
noncomputable abbrev Q114.gt := fun (answer : ℝ) => answer = (1 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q114
find (answer : ℝ) s.t.
: (answer = ∫ x in (2)..4, sqrt (log (9 - x)) / (sqrt (log (9 - x)) + sqrt (log (x + 3))))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q114
find_all (answer : ℝ) iff
: (answer = ∫ x in (2)..4, sqrt (log (9 - x)) / (sqrt (log (9 - x)) + sqrt (log (x + 3))))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
