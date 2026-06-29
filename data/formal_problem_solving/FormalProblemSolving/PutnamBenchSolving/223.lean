import FormalProblemSolving.Basic
open Topology MvPolynomial Filter Set

/-
## Problem
Let $d_n$ be the determinant of the $n \times n$ matrix whose entries, from left to right and then from top to bottom, are $\cos 1, \cos 2, \dots, \cos n^2$. (For example,\[ d_3 = \left|\begin{matrix} \cos 1 & \cos 2 & \cos 3 \\ \cos 4 & \cos 5 & \cos 6 \\  \cos 7 & \cos 8 & \cos 9 \end{matrix} \right|. \]The argument of $\cos$ is always in radians, not degrees.) Evaluate $\lim_{n\to\infty} d_n$.

## Answer
0
-/

-- # Formal Answer
noncomputable abbrev Q223.gt := fun (answer : ℝ) => answer = (0 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q223
find (answer : ℝ) s.t.
  (cos_matrix : (n : ℕ) → Matrix (Fin n) (Fin n) ℝ)
  (hM : ∀ n : ℕ, ∀ i j : Fin n, (cos_matrix n) i j = Real.cos (1 + n * i + j))
: (Tendsto (fun n => (cos_matrix n).det) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q223
find_all (answer : ℝ) iff
  (cos_matrix : (n : ℕ) → Matrix (Fin n) (Fin n) ℝ)
  (hM : ∀ n : ℕ, ∀ i j : Fin n, (cos_matrix n) i j = Real.cos (1 + n * i + j))
: (Tendsto (fun n => (cos_matrix n).det) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
