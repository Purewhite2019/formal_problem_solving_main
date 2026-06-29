import FormalProblemSolving.Basic
open Topology Filter Set Matrix

/-
## Problem
Let $F_m$ be the $m$th Fibonacci number, defined by $F_1 = F_2 = 1$ and $F_m = F_{m-1} + F_{m-2}$ for all $m \geq 3$. Let $p(x)$ be the polynomial of degree $1008$ such that $p(2n + 1) = F_{2n+1}$ for $n = 0,1,2,\ldots,1008$. Find integers $j$ and $k$ such that $p(2019) = F_j - F_k$.

## Answer
⟨2019, 1010⟩
-/

-- # Formal Answer
noncomputable abbrev Q288.gt := fun (answer : ℕ × ℕ) => answer = (((2019 : ℕ), (1010 : ℕ)) : ℕ × ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q288
find (answer : ℕ × ℕ) s.t.
  (F : ℕ → ℤ)
  (P : Polynomial ℝ)
  (hF : ∀ x, x ≥ 1 → F (x + 2) = F (x + 1) + F x)
  (F12 : F 1 = 1 ∧ F 2 = 1)
  (Pdeg : Polynomial.degree P = 1008)
  (hp : ∀ n : ℕ, (n ≤ 1008) → P.eval (2 * n + 1 : ℝ) = F (2 * n + 1))
: (∀ j k : ℕ, (P.eval 2019 = F j - F k) ↔ ⟨j, k⟩ = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q288
find_all (answer : ℕ × ℕ) iff
  (F : ℕ → ℤ)
  (P : Polynomial ℝ)
  (hF : ∀ x, x ≥ 1 → F (x + 2) = F (x + 1) + F x)
  (F12 : F 1 = 1 ∧ F 2 = 1)
  (Pdeg : Polynomial.degree P = 1008)
  (hp : ∀ n : ℕ, (n ≤ 1008) → P.eval (2 * n + 1 : ℝ) = F (2 * n + 1))
: (∀ j k : ℕ, (P.eval 2019 = F j - F k) ↔ ⟨j, k⟩ = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
