import FormalProblemSolving.Basic
open Topology Filter Nat

/-
## Problem
Let $R$ be the region consisting of all triples $(x,y,z)$ of nonnegative real numbers satisfying $x+y+z \leq 1$. Let $w=1-x-y-z$. Express the value of the triple integral $\iiint_R x^1y^9z^8w^4\,dx\,dy\,dz$ in the form $a!b!c!d!/n!$, where $a$, $b$, $c$, $d$, and $n$ are positive integers.

## Answer
(1, 9, 8, 4, 25)
-/

-- # Formal Answer
noncomputable abbrev Q87.gt := fun (answer : ℕ × ℕ × ℕ × ℕ × ℕ) => answer =
  (((1 : ℕ), (((9 : ℕ), (((8 : ℕ), (((4 : ℕ), (25 : ℕ)) : ℕ × ℕ)) : ℕ × ℕ × ℕ)) : ℕ × ℕ × ℕ × ℕ)) : ℕ × ℕ × ℕ × ℕ × ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q87
find (answer : ℕ × ℕ × ℕ × ℕ × ℕ) s.t.
  (R : Set (Fin 3 → ℝ))
  (w : (Fin 3 → ℝ) → ℝ)
  (hR : R = {p | (∀ i : Fin 3, p i ≥ 0) ∧ p 0 + p 1 + p 2 ≤ 1})
  (hw : ∀ p, w p = 1 - p 0 - p 1 - p 2)
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (d : ℕ)
  (n : ℕ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ n > 0)
  (h_integral : (∫ p in R, (p 0) ^ 1 * (p 1) ^ 9 * (p 2) ^ 8 * (w p) ^ 4 = ((a)! * (b)! * (c)! * (d)! : ℝ) / (n)!))
: (answer = (a, b, c, d, n))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q87
find_all (answer : ℕ × ℕ × ℕ × ℕ × ℕ) iff
  (R : Set (Fin 3 → ℝ))
  (w : (Fin 3 → ℝ) → ℝ)
  (hR : R = {p | (∀ i : Fin 3, p i ≥ 0) ∧ p 0 + p 1 + p 2 ≤ 1})
  (hw : ∀ p, w p = 1 - p 0 - p 1 - p 2)
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)
  (d : ℕ)
  (n : ℕ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ n > 0)
  (h_integral : (∫ p in R, (p 0) ^ 1 * (p 1) ^ 9 * (p 2) ^ 8 * (w p) ^ 4 = ((a)! * (b)! * (c)! * (d)! : ℝ) / (n)!))
: (answer = (a, b, c, d, n))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
