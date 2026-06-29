import FormalProblemSolving.Basic

/-
## Problem
What is the units (i.e., rightmost) digit of
\[
\left\lfloor \frac{10^{20000}}{10^{100}+3}\right\rfloor ?
\]

## Answer
3
-/

-- # Formal Answer
noncomputable abbrev Q102.gt := fun (answer : ℕ) => answer = (3 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q102
find (answer : ℕ) s.t.
: (Nat.floor ((10 ^ 20000 : ℝ) / (10 ^ 100 + 3)) % 10 = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q102
find_all (answer : ℕ) iff
: (Nat.floor ((10 ^ 20000 : ℝ) / (10 ^ 100 + 3)) % 10 = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
