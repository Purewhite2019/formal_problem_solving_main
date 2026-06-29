import FormalProblemSolving.Basic
open Topology Filter Set Polynomial Function

/-
## Problem
Let $E(n)$ be the greatest integer $k$ such that $5^k$ divides $1^1 2^2 3^3 \cdots n^n$. Find $\lim_{n \rightarrow \infty} \frac{E(n)}{n^2}$.

## Answer
$\frac{1}{8}$
-/

-- # Formal Answer
noncomputable abbrev Q67.gt := fun (answer : ℝ) => answer = (1 / 8 : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q67
find (answer : ℝ) s.t.
  (P : ℕ → ℕ → Prop)
  (hP : ∀ n k, P n k ↔ 5^k ∣ ∏ m in Finset.Icc 1 n, (m^m : ℤ))
  (E : ℕ → ℕ)
  (hE : ∀ n ∈ Ici 1, P n (E n) ∧ ∀ k : ℕ, P n k → k ≤ E n)
: (Tendsto (fun n : ℕ => ((E n) : ℝ)/n^2) atTop (𝓝 answer))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q67
find_all (answer : ℝ) iff
  (P : ℕ → ℕ → Prop)
  (hP : ∀ n k, P n k ↔ 5^k ∣ ∏ m in Finset.Icc 1 n, (m^m : ℤ))
  (E : ℕ → ℕ)
  (hE : ∀ n ∈ Ici 1, P n (E n) ∧ ∀ k : ℕ, P n k → k ≤ E n)
: (Tendsto (fun n : ℕ => ((E n) : ℝ)/n^2) atTop (𝓝 answer))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
