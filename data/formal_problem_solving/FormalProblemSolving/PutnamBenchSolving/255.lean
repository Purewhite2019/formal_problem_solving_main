import FormalProblemSolving.Basic

/-
## Problem
Let $a_0=1$, $a_1=2$, and $a_n=4a_{n-1}-a_{n-2}$ for $n \geq 2$. Find an odd prime factor of $a_{2015}$.

## Answer
181
-/

-- # Formal Answer
noncomputable abbrev Q255.gt := fun (answer : ℕ) => answer = (181 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q255
find (answer : ℕ) s.t.
  (a : ℕ → ℤ)
  (abase : a 0 = 1 ∧ a 1 = 2)
  (arec : ∀ n ≥ 2, a n = 4 * a (n - 1) - a (n - 2))
: (Odd answer ∧ answer.Prime ∧ ((answer : ℤ) ∣ a 2015))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q255
find_all (answer : ℕ) iff
  (a : ℕ → ℤ)
  (abase : a 0 = 1 ∧ a 1 = 2)
  (arec : ∀ n ≥ 2, a n = 4 * a (n - 1) - a (n - 2))
: (Odd answer ∧ answer.Prime ∧ ((answer : ℤ) ∣ a 2015))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
