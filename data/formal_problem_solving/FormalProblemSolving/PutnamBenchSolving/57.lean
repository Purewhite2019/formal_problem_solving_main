import FormalProblemSolving.Basic

/-
## Problem
Let $x_1, x_2, x_3, \dots$ be a sequence of nonzero real numbers such that $$x_n = \frac{x_{n-2}x_{n-1}}{2x_{n-2}-x_{n-1}}$$ for all $n \ge 3$. For which real values of $x_1$ and $x_2$ does $x_n$ attain integer values for infinitely many $n$?

## Answer
fun (a, b) => ∃ m : ℤ, a = m ∧ b = m
-/

-- # Formal Answer
noncomputable abbrev Q57.gt := fun (x : ℕ → ℝ) (answer : (ℝ × ℝ) → Prop) => answer = fun (x : ℝ × ℝ) =>
  match x with
  | ((a, b) : ℝ × ℝ) => ∃ (m : ℤ), a = (↑m : ℝ) ∧ b = (↑m : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q57
  (x : ℕ → ℝ)
find (answer : (ℝ × ℝ) → Prop) s.t.
  (hx : ∀ n : ℕ, x n ≠ 0 ∧ (n ≥ 3 → x n = (x (n - 2))*(x (n - 1))/(2*(x (n - 2)) - (x (n - 1)))))
: ((∀ m : ℕ, ∃ n : ℕ, n > m ∧ ∃ a : ℤ, a = x n) ↔ answer (x 1, x 2))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q57
  (x : ℕ → ℝ)
find_all (answer : (ℝ × ℝ) → Prop) iff
  (hx : ∀ n : ℕ, x n ≠ 0 ∧ (n ≥ 3 → x n = (x (n - 2))*(x (n - 1))/(2*(x (n - 2)) - (x (n - 1)))))
: ((∀ m : ℕ, ∃ n : ℕ, n > m ∧ ∃ a : ℤ, a = x n) ↔ answer (x 1, x 2))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
