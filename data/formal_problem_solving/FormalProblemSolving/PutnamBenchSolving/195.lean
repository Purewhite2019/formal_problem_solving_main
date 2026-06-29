import FormalProblemSolving.Basic
open Nat Topology Filter

/-
## Problem
Let $n$ be a positive integer, $n \ge 2$, and put $\theta = 2 \pi / n$. Define points $P_k = (k,0)$ in the $xy$-plane, for $k = 1, 2, \dots, n$. Let $R_k$ be the map that rotates the plane counterclockwise by the angle $\theta$ about the point $P_k$. Let $R$ denote the map obtained by applying, in order, $R_1$, then $R_2, \dots$, then $R_n$. For an arbitrary point $(x,y)$, find, and simplify, the coordinates of $R(x,y)$.

## Answer
fun n z ↦ z + n
-/

-- # Formal Answer
noncomputable abbrev Q195.gt := fun (n : ℕ) (answer : ℕ → ℂ → ℂ) => answer = fun (n : ℕ) (z : ℂ) => (z + (↑n : ℂ) : ℂ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q195
  (n : ℕ)
find (answer : ℕ → ℂ → ℂ) s.t.
  (nge2 : n ≥ 2)
  (R : ℕ → ℂ → ℂ)
  (Rk : ℕ → ℂ → ℂ)
  (hR : R 0 = id ∧ ∀ k : ℕ, R (k + 1) = Rk (k + 1) ∘ R k)
  (hRk : Rk = fun (k : ℕ) (Q : ℂ) ↦ k + Complex.exp (Complex.I * 2 * Real.pi / n) * (Q - k))
: (R n = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q195
  (n : ℕ)
find_all (answer : ℕ → ℂ → ℂ) iff
  (nge2 : n ≥ 2)
  (R : ℕ → ℂ → ℂ)
  (Rk : ℕ → ℂ → ℂ)
  (hR : R 0 = id ∧ ∀ k : ℕ, R (k + 1) = Rk (k + 1) ∘ R k)
  (hRk : Rk = fun (k : ℕ) (Q : ℂ) ↦ k + Complex.exp (Complex.I * 2 * Real.pi / n) * (Q - k))
: (R n = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
