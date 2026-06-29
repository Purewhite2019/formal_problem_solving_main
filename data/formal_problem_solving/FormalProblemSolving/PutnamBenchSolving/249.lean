import FormalProblemSolving.Basic
open Topology Filter Nat

/-
## Problem
Let $A$ be the $n \times n$ matrix whose entry in the $i$-th row and $j$-th column is $\frac{1}{\min(i,j)}$ for $1 \leq i,j \leq n$. Compute $\det(A)$.

## Answer
(fun n : ℕ => (-1) ^ (n - 1) / ((n - 1)! * (n)!))
-/

-- # Formal Answer
noncomputable abbrev Q249.gt := fun (n : ℕ) (answer : ℕ → ℝ) => answer = fun (n : ℕ) =>
  (((-1 : ℝ) ^ (n - (1 : ℕ) : ℕ) : ℝ) / ((↑((n - (1 : ℕ) : ℕ)! : ℕ) : ℝ) * (↑(n ! : ℕ) : ℝ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q249
  (n : ℕ)
find (answer : ℕ → ℝ) s.t.
  (A : Matrix (Fin n) (Fin n) ℝ)
  (npos : n > 0)
  (hA : ∀ i j : Fin n, A i j = 1 / min (i.1 + 1 : ℚ) (j.1 + 1))
: (A.det = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q249
  (n : ℕ)
find_all (answer : ℕ → ℝ) iff
  (A : Matrix (Fin n) (Fin n) ℝ)
  (npos : n > 0)
  (hA : ∀ i j : Fin n, A i j = 1 / min (i.1 + 1 : ℚ) (j.1 + 1))
: (A.det = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
