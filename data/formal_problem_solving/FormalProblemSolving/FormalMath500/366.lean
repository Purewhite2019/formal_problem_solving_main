import FormalProblemSolving.Basic
open BigOperators Real Nat Topology

/-
## Problem
The complex numbers $\alpha_1, \alpha_2, \alpha_3$, and $\alpha_4$ are the four distinct roots of the equation $x^4+2x^3+2=0$. Determine the unordered set \[
\{\alpha_1\alpha_2 + \alpha_3\alpha_4, \alpha_1\alpha_3 + \alpha_2\alpha_4, \alpha_1\alpha_4 + \alpha_2\alpha_3\}.
\]

## Answer
\{1\pm\sqrt{5},-2\}
-/

-- # Formal Answer
noncomputable abbrev Q366.gt := fun (answer : Set ℂ) => answer =
  (Insert.insert ((1 : ℂ) + (↑(√(5 : ℝ) : ℝ) : ℂ) : ℂ)
      (Insert.insert ((1 : ℂ) - (↑(√(5 : ℝ) : ℝ) : ℂ) : ℂ) ({(-2 : ℂ)} : Set ℂ) : Set ℂ) :
    Set ℂ)

-- # Formal Problem (FPS)
namespace FormalMath500_FPS
open scoped FPS

problem Q366
find (answer : Set ℂ) s.t.
  (a : ℂ)
  (b : ℂ)
  (c : ℂ)
  (d : ℂ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (hr : a ^ 4 + 2 * a ^ 3 + 2 = 0 ∧ b ^ 4 + 2 * b ^ 3 + 2 = 0 ∧ c ^ 4 + 2 * c ^ 3 + 2 = 0 ∧ d ^ 4 + 2 * d ^ 3 + 2 = 0)
: (answer = {a * b + c * d, a * c + b * d, a * d + b * c})
:= solve
-- Placeholder
all_goals sorry

end FormalMath500_FPS

-- # Formal Problem (D-FPS)
namespace FormalMath500_DFPS
open scoped DFPS

problem Q366
find_all (answer : Set ℂ) iff
  (a : ℂ)
  (b : ℂ)
  (c : ℂ)
  (d : ℂ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (hr : a ^ 4 + 2 * a ^ 3 + 2 = 0 ∧ b ^ 4 + 2 * b ^ 3 + 2 = 0 ∧ c ^ 4 + 2 * c ^ 3 + 2 = 0 ∧ d ^ 4 + 2 * d ^ 3 + 2 = 0)
: (answer = {a * b + c * d, a * c + b * d, a * d + b * c})
:= solve
case Answer := sorry
all_goals sorry

end FormalMath500_DFPS
