import FormalProblemSolving.Basic
open Set MvPolynomial

/-
## Problem
After each play of a certain game of solitaire, the player receives either $a$ or $b$ points, where $a$ and $b$ are positive integers with $a > b$; scores accumulate from play to play. If there are $35$ unattainable scores, one of which is $58$, find $a$ and $b$.

## Answer
(11, 8)
-/

-- # Formal Answer
noncomputable abbrev Q21.gt := fun (answer : ℤ × ℤ) => answer = (((11 : ℤ), (8 : ℤ)) : ℤ × ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q21
find (answer : ℤ × ℤ) s.t.
  (a : ℤ)
  (b : ℤ)
  (hab : a > 0 ∧ b > 0 ∧ a > b)
  (pab : ℤ → ℤ → Prop)
  (hpab : ∀ x y, pab x y ↔
    {s : ℕ | ¬∃ m n : ℕ, m*x + n*y = s}.ncard = 35 ∧
    ¬∃ m n : ℕ, m*x + n*y = 58)
: (pab a b ↔ a = answer.1 ∧ b = answer.2)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q21
find_all (answer : ℤ × ℤ) iff
  (a : ℤ)
  (b : ℤ)
  (hab : a > 0 ∧ b > 0 ∧ a > b)
  (pab : ℤ → ℤ → Prop)
  (hpab : ∀ x y, pab x y ↔
    {s : ℕ | ¬∃ m n : ℕ, m*x + n*y = s}.ncard = 35 ∧
    ¬∃ m n : ℕ, m*x + n*y = 58)
: (pab a b ↔ a = answer.1 ∧ b = answer.2)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
