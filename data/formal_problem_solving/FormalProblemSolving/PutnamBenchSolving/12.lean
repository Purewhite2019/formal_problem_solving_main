import FormalProblemSolving.Basic
open Nat Topology Filter

/-
## Problem
Given real numbers $\{a_i\}$ and $\{b_i\}$, ($i=1,2,3,4$), such that $a_1b_2-a_2b_1 \neq 0$. Consider the set of all solutions $(x_1,x_2,x_3,x_4)$ of the simultaneous equations $a_1x_1+a_2x_2+a_3x_3+a_4x_4=0$ and $b_1x_1+b_2x_2+b_3x_3+b_4x_4=0$, for which no $x_i$ ($i=1,2,3,4$) is zero. Each such solution generates a $4$-tuple of plus and minus signs $(\text{signum }x_1,\text{signum }x_2,\text{signum }x_3,\text{signum }x_4)$. Determine, with a proof, the maximum number of distinct $4$-tuples possible.

## Answer
8
-/

-- # Formal Answer
noncomputable abbrev Q12.gt := fun (answer : ℕ) => answer = (8 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q12
find (answer : ℕ) s.t.
  (abneq0 : (Fin 4 → ℝ) → (Fin 4 → ℝ) → Prop)
  (habneq0 : abneq0 = (fun a b : Fin 4 → ℝ => a 0 * b 1 - a 1 * b 0 ≠ 0))
  (numtuples : (Fin 4 → ℝ) → (Fin 4 → ℝ) → ℕ)
  (hnumtuples : ∀ a b : Fin 4 → ℝ, numtuples a b = {s : Fin 4 → ℝ | ∃ x : Fin 4 → ℝ, (∀ i : Fin 4, x i ≠ 0) ∧ (∑ i : Fin 4, a i * x i) = 0 ∧ (∑ i : Fin 4, b i * x i) = 0 ∧ (∀ i : Fin 4, s i = Real.sign (x i))}.encard)
: ((∃ a b : Fin 4 → ℝ, abneq0 a b ∧ numtuples a b = answer) ∧ (∀ a b : Fin 4 → ℝ, abneq0 a b → numtuples a b ≤ answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q12
find_all (answer : ℕ) iff
  (abneq0 : (Fin 4 → ℝ) → (Fin 4 → ℝ) → Prop)
  (habneq0 : abneq0 = (fun a b : Fin 4 → ℝ => a 0 * b 1 - a 1 * b 0 ≠ 0))
  (numtuples : (Fin 4 → ℝ) → (Fin 4 → ℝ) → ℕ)
  (hnumtuples : ∀ a b : Fin 4 → ℝ, numtuples a b = {s : Fin 4 → ℝ | ∃ x : Fin 4 → ℝ, (∀ i : Fin 4, x i ≠ 0) ∧ (∑ i : Fin 4, a i * x i) = 0 ∧ (∑ i : Fin 4, b i * x i) = 0 ∧ (∀ i : Fin 4, s i = Real.sign (x i))}.encard)
: ((∃ a b : Fin 4 → ℝ, abneq0 a b ∧ numtuples a b = answer) ∧ (∀ a b : Fin 4 → ℝ, abneq0 a b → numtuples a b ≤ answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
