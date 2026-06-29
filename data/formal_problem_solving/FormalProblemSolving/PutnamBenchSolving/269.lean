import FormalProblemSolving.Basic
open Real Set Nat

/-
## Problem
Let $A$ be a $2n \times 2n$ matrix, with entries chosen independently at random. Every entry is chosen to be $0$ or $1$, each with probability $1/2$. Find the expected value of $\det(A-A^t)$ (as a function of $n$), where $A^t$ is the transpose of $A$.

## Answer
(fun n : ℕ => (2 * n)! / (4 ^ n * (n)!))
-/

-- # Formal Answer
noncomputable abbrev Q269.gt := fun (n : ℕ) (answer : ℕ → ℝ) => answer = fun (n : ℕ) => ((↑(((2 : ℕ) * n : ℕ)! : ℕ) : ℝ) / (((4 : ℝ) ^ n : ℝ) * (↑(n ! : ℕ) : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q269
  (n : ℕ)
find (answer : ℕ → ℝ) s.t.
  (npos : n ≥ 1)
  (mats01 : Set (Matrix (Fin (2 * n)) (Fin (2 * n)) ℝ))
  (hmats01 : mats01 = {A | ∀ i j : Fin (2 * n), A i j = 0 ∨ A i j = 1})
: ((∑' A : mats01, (A.1 - (Matrix.transpose A)).det) / mats01.ncard = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q269
  (n : ℕ)
find_all (answer : ℕ → ℝ) iff
  (npos : n ≥ 1)
  (mats01 : Set (Matrix (Fin (2 * n)) (Fin (2 * n)) ℝ))
  (hmats01 : mats01 = {A | ∀ i j : Fin (2 * n), A i j = 0 ∨ A i j = 1})
: ((∑' A : mats01, (A.1 - (Matrix.transpose A)).det) / mats01.ncard = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
