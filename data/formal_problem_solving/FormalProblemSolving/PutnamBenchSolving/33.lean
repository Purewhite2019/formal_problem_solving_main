import FormalProblemSolving.Basic
open Polynomial

/-
## Problem
If an integer $n$ can be written as the sum of two triangular numbers (that is, $n = \frac{a^2 + a}{2} + \frac{b^2 + b}{2}$ for some integers $a$ and $b$), express $4n + 1$ as the sum of the squares of two integers $x$ and $y$, giving $x$ and $y$ in terms of $a$ and $b$. Also, show that if $4n + 1 = x^2 + y^2$ for some integers $x$ and $y$, then $n$ can be written as the sum of two triangular numbers.

## Answer
(fun (a, b) => a + b + 1, fun (a, b) => a - b)
-/

-- # Formal Answer
noncomputable abbrev Q33.gt := fun (answer : ((ℤ × ℤ) → ℤ) × ((ℤ × ℤ) → ℤ)) => answer =
  ((fun (x : ℤ × ℤ) =>
      match x with
      | ((a, b) : ℤ × ℤ) => ((a + b : ℤ) + (1 : ℤ) : ℤ),
      fun (x : ℤ × ℤ) =>
      match x with
      | ((a, b) : ℤ × ℤ) => (a - b : ℤ)) :
    (ℤ × ℤ → ℤ) × (ℤ × ℤ → ℤ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q33
find (answer : ((ℤ × ℤ) → ℤ) × ((ℤ × ℤ) → ℤ)) s.t.
  (nab : (ℤ × ℤ × ℤ) → Prop)
  (nxy : (ℤ × ℤ × ℤ) → Prop)
  (hnab : nab = fun (n, a, b) => n = (a^2 + (a : ℚ))/2 + (b^2 + (b : ℚ))/2)
  (hnxy : nxy = fun (n, x, y) => 4*n + 1 = x^2 + y^2)
: ((∀ n a b : ℤ, nab (n, a, b) → nxy (n, answer.1 (a, b), answer.2 (a, b))) ∧ ∀ n : ℤ, (∃ x y : ℤ, nxy (n, x, y)) → ∃ a b : ℤ, nab (n, a, b))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q33
find_all (answer : ((ℤ × ℤ) → ℤ) × ((ℤ × ℤ) → ℤ)) iff
  (nab : (ℤ × ℤ × ℤ) → Prop)
  (nxy : (ℤ × ℤ × ℤ) → Prop)
  (hnab : nab = fun (n, a, b) => n = (a^2 + (a : ℚ))/2 + (b^2 + (b : ℚ))/2)
  (hnxy : nxy = fun (n, x, y) => 4*n + 1 = x^2 + y^2)
: ((∀ n a b : ℤ, nab (n, a, b) → nxy (n, answer.1 (a, b), answer.2 (a, b))) ∧ ∀ n : ℤ, (∃ x y : ℤ, nxy (n, x, y)) → ∃ a b : ℤ, nab (n, a, b))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
