import FormalProblemSolving.Basic
open Polynomial

/-
## Problem
Find all integers $n$ with $n \geq 4$ for which there exists a sequence of distinct real numbers $x_1,\dots,x_n$ such that each of the sets $\{x_1,x_2,x_3\},\{x_2,x_3,x_4\},\dots,\{x_{n-2},x_{n-1},x_n\},\{x_{n-1},x_n,x_1\}$, and $\{x_n,x_1,x_2\}$ forms a $3$-term arithmetic progression when arranged in increasing order.

## Answer
{n : ℕ | 3 ∣ n ∧ n ≥ 9}
-/

-- # Formal Answer
noncomputable abbrev Q311.gt := fun (n : ℕ) (answer : Set ℕ) => answer = ({n : ℕ | (3 : ℕ) ∣ n ∧ n ≥ (9 : ℕ)} : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q311
  (n : ℕ)
find (answer : Set ℕ) s.t.
  (hn : n ≥ 4)
  (ap3 : ℝ → ℝ → ℝ → Prop)
  (xprog : (ℕ → ℝ) → Prop)
  (hap3 : ∀ x0 x1 x2, ap3 x0 x1 x2 ↔ ∀ o0 o1 o2 : ℝ, (o0 < o1 ∧ o1 < o2 ∧ ({o0, o1, o2} : Set ℝ) = {x0, x1, x2}) → (o1 - o0 = o2 - o1))
  (hxprog : ∀ x, xprog x ↔
    ((∀ i j : Fin n, i.1 ≠ j.1 → x i.1 ≠ x j.1) ∧ (∀ i : Fin n, ap3 (x i.1) (x ((i.1 + 1) % n)) (x ((i.1 + 2) % n)))))
: ((∃ x, xprog x) ↔ n ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q311
  (n : ℕ)
find_all (answer : Set ℕ) iff
  (hn : n ≥ 4)
  (ap3 : ℝ → ℝ → ℝ → Prop)
  (xprog : (ℕ → ℝ) → Prop)
  (hap3 : ∀ x0 x1 x2, ap3 x0 x1 x2 ↔ ∀ o0 o1 o2 : ℝ, (o0 < o1 ∧ o1 < o2 ∧ ({o0, o1, o2} : Set ℝ) = {x0, x1, x2}) → (o1 - o0 = o2 - o1))
  (hxprog : ∀ x, xprog x ↔
    ((∀ i j : Fin n, i.1 ≠ j.1 → x i.1 ≠ x j.1) ∧ (∀ i : Fin n, ap3 (x i.1) (x ((i.1 + 1) % n)) (x ((i.1 + 2) % n)))))
: ((∃ x, xprog x) ↔ n ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
