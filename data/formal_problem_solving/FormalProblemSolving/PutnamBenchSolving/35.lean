import FormalProblemSolving.Basic
open Polynomial

/-
## Problem
If $a$, $b$, and $c$ are real numbers satisfying $0 < a < b < c$, at what points in the set $$\{(x, y, z) \in \mathbb{R}^3 : x^b + y^b + z^b = 1, x \ge 0, y \ge 0, z \ge 0\}$$ does $f(x, y, z) = x^a + y^b + z^c$ attain its maximum and minimum?

## Answer
(fun (a, b, c) => ((a/b)^(1/(b - a)), (1 - ((a/b)^(1/(b - a)))^b)^(1/b), 0), fun (a, b, c) => (0, (1 - ((b/c)^(1/(c - b)))^b)^(1/b), (b/c)^(1/(c - b))))
-/

-- # Formal Answer
noncomputable abbrev Q35.gt := fun (a : ℝ) (b : ℝ) (c : ℝ) (answer : ((ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ)) × ((ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ))) => answer =
  ((fun (x : ℝ × ℝ × ℝ) =>
      match x with
      | ((a, ((b, c) : ℝ × ℝ)) : ℝ × ℝ × ℝ) =>
        ((((a / b : ℝ) ^ ((1 : ℝ) / (b - a : ℝ) : ℝ) : ℝ),
            (((((1 : ℝ) - (((a / b : ℝ) ^ ((1 : ℝ) / (b - a : ℝ) : ℝ) : ℝ) ^ b : ℝ) : ℝ) ^ ((1 : ℝ) / b : ℝ) : ℝ),
                (0 : ℝ)) :
              ℝ × ℝ)) :
          ℝ × ℝ × ℝ),
      fun (x : ℝ × ℝ × ℝ) =>
      match x with
      | ((a, ((b, c) : ℝ × ℝ)) : ℝ × ℝ × ℝ) =>
        (((0 : ℝ),
            (((((1 : ℝ) - (((b / c : ℝ) ^ ((1 : ℝ) / (c - b : ℝ) : ℝ) : ℝ) ^ b : ℝ) : ℝ) ^ ((1 : ℝ) / b : ℝ) : ℝ),
                ((b / c : ℝ) ^ ((1 : ℝ) / (c - b : ℝ) : ℝ) : ℝ)) :
              ℝ × ℝ)) :
          ℝ × ℝ × ℝ)) :
    (ℝ × ℝ × ℝ → ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ → ℝ × ℝ × ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q35
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
find (answer : ((ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ)) × ((ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ))) s.t.
  (hi : 0 < a ∧ a < b ∧ b < c)
  (P : (ℝ × ℝ × ℝ) → Prop)
  (f : (ℝ × ℝ × ℝ) → ℝ)
  (hP : P = fun (x, y, z) => x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^b + y^b + z^b = 1)
  (hf : f = fun (x, y, z) => x^a + y^b + z^c)
: ((P (answer.1 (a, b, c)) ∧ ∀ x y z : ℝ, P (x, y, z) →
f (x, y, z) ≤ f (answer.1 (a, b, c))) ∧
(P (answer.2 (a, b, c)) ∧ ∀ x y z : ℝ, P (x, y, z) →
f (x, y, z) ≥ f (answer.2 (a, b, c))))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q35
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
find_all (answer : ((ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ)) × ((ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ))) iff
  (hi : 0 < a ∧ a < b ∧ b < c)
  (P : (ℝ × ℝ × ℝ) → Prop)
  (f : (ℝ × ℝ × ℝ) → ℝ)
  (hP : P = fun (x, y, z) => x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^b + y^b + z^b = 1)
  (hf : f = fun (x, y, z) => x^a + y^b + z^c)
: ((P (answer.1 (a, b, c)) ∧ ∀ x y z : ℝ, P (x, y, z) →
f (x, y, z) ≤ f (answer.1 (a, b, c))) ∧
(P (answer.2 (a, b, c)) ∧ ∀ x y z : ℝ, P (x, y, z) →
f (x, y, z) ≥ f (answer.2 (a, b, c))))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
