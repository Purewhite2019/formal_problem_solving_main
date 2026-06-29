import FormalProblemSolving.Basic
open Topology Filter Nat Function Polynomial

/-
## Problem
Let $D_n$ denote the value of the $(n-1) \times (n-1)$ determinant
\[
\left[
\begin{array}{cccccc}
3 & 1 & 1 & 1 & \cdots & 1 \\
1 & 4 & 1 & 1 & \cdots & 1 \\
1 & 1 & 5 & 1 & \cdots & 1 \\
1 & 1 & 1 & 6 & \cdots & 1 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
1 & 1 & 1 & 1 & \cdots & n+1
\end{array}
\right].
\]
Is the set $\left\{ \frac{D_n}{n!} \right\}_{n \geq 2}$ bounded?

## Answer
False
-/

-- # Formal Answer
noncomputable abbrev Q151.gt := fun (answer : Prop) => answer = False

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q151
find (answer : Prop) s.t.
  (D : ℕ → ℚ)
  (hD : ∀ n, D n = Matrix.det (fun i j : Fin (n - 1) ↦ ite (i = j) ((i : ℕ) + 3 : ℚ) 1))
: (answer ↔ (Bornology.IsBounded {x | ∃ n ≥ 2, D n / factorial n = x}))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q151
find_all (answer : Prop) iff
  (D : ℕ → ℚ)
  (hD : ∀ n, D n = Matrix.det (fun i j : Fin (n - 1) ↦ ite (i = j) ((i : ℕ) + 3 : ℚ) 1))
: (answer ↔ (Bornology.IsBounded {x | ∃ n ≥ 2, D n / factorial n = x}))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
