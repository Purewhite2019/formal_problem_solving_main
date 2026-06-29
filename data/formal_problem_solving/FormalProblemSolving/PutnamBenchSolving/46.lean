import FormalProblemSolving.Basic

/-
## Problem
Find all real solutions $(a, b, c, d)$ to the equations $a + b + c = d$, $\frac{1}{a} + \frac{1}{b} + \frac{1}{c} = \frac{1}{d}$.

## Answer
$d = a \land b = -c \lor d = b \land a = -c \lor d = c \land a = -b$
-/

-- # Formal Answer
noncomputable abbrev Q46.gt := fun (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (answer : ℝ → ℝ → ℝ → ℝ → Prop) => answer = fun (a b c d : ℝ) => d = a ∧ b = (-c : ℝ) ∨ d = b ∧ a = (-c : ℝ) ∨ d = c ∧ a = (-b : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q46
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
find (answer : ℝ → ℝ → ℝ → ℝ → Prop) s.t.
: (answer a b c d ↔
    a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 → (a + b + c = d ∧ 1 / a + 1 / b + 1 / c = 1 / d))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q46
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (d : ℝ)
find_all (answer : ℝ → ℝ → ℝ → ℝ → Prop) iff
: (answer a b c d ↔
    a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 → (a + b + c = d ∧ 1 / a + 1 / b + 1 / c = 1 / d))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
