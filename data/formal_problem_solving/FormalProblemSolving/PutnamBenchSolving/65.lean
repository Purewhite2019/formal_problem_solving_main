import FormalProblemSolving.Basic

/-
## Problem
For which real numbers $a$ does the sequence defined by the initial condition $u_0=a$ and the recursion $u_{n+1}=2u_n-n^2$ have $u_n>0$ for all $n \geq 0$? (Express the answer in the simplest form.)

## Answer
$\{a : \mathbb{R} \mid a \geq 3\}$
-/

-- # Formal Answer
noncomputable abbrev Q65.gt := fun (a : ℝ) (answer : Set ℝ) => answer = ({a : ℝ | a ≥ (3 : ℝ)} : Set ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q65
  (a : ℝ)
find (answer : Set ℝ) s.t.
  (u : ℕ → ℝ)
  (hu : u 0 = a ∧ (∀ n : ℕ, u (n + 1) = 2 * u n - n ^ 2))
: ((∀ n : ℕ, u n > 0) ↔ a ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q65
  (a : ℝ)
find_all (answer : Set ℝ) iff
  (u : ℕ → ℝ)
  (hu : u 0 = a ∧ (∀ n : ℕ, u (n + 1) = 2 * u n - n ^ 2))
: ((∀ n : ℕ, u n > 0) ↔ a ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
