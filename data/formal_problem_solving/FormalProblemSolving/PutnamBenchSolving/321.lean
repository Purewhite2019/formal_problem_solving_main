import FormalProblemSolving.Basic
open ProbabilityTheory Set

/-
## Problem
A sequence $y_1, y_2, \ldots, y_k$ of real numbers is called zigzag if $k = 1$, or if $y_2 - y_1, y_3 - y_2, \ldots, y_k - y_{k-1}$ are nonzero and alternate in sign. Let $X_1, X_2,\ldots, X_n$ be chosen independently from the uniform distribution on $[0,1]$. Let $a(X_1, X_2, \ldots, X_n)$ be the largest value of $k$ for which there exists an increasing sequence of integers $i_1, i_2, \ldots, i_k$ such that $X_{i_1}, X_{i_2}, \ldots, X_{i_k}$ is zigzag. Find the expected value of $a(X_1, X_2, \ldots, X_n)$ for $n \ge 2$.

## Answer
(2n + 2) / 3
-/

-- # Formal Answer
noncomputable abbrev Q321.gt := fun (n : ℕ) (answer : ℕ → ℝ) => answer = fun (n : ℕ) => ((((2 : ℝ) * (↑n : ℝ) : ℝ) + (2 : ℝ) : ℝ) / (3 : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q321
  (n : ℕ)
find (answer : ℕ → ℝ) s.t.
  (IsZigZag : {k : ℕ} → (Fin k → ℝ) → Prop)
  (IsZigZag_def : ∀ (k : ℕ) [NeZero k] (y : Fin k → ℝ),
      IsZigZag y ↔ k = 1 ∨ ((∀ i, i + 1 < k → y (i + 1) ≠ y i)) ∧
        (∀ i, i + 2 < k → (y (i + 2) < y (i + 1) ↔ y i < y (i + 1))))
  (hn : 2 ≤ n)
  (a : (Fin n → Icc (0 : ℝ) 1) → ℕ)
  (ha : ∀ x, IsGreatest {k | ∃ i : Fin k ↪o Fin n, IsZigZag ((↑) ∘ x ∘ i)} (a x))
: (𝔼[(↑) ∘ a] = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q321
  (n : ℕ)
find_all (answer : ℕ → ℝ) iff
  (IsZigZag : {k : ℕ} → (Fin k → ℝ) → Prop)
  (IsZigZag_def : ∀ (k : ℕ) [NeZero k] (y : Fin k → ℝ),
      IsZigZag y ↔ k = 1 ∨ ((∀ i, i + 1 < k → y (i + 1) ≠ y i)) ∧
        (∀ i, i + 2 < k → (y (i + 2) < y (i + 1) ↔ y i < y (i + 1))))
  (hn : 2 ≤ n)
  (a : (Fin n → Icc (0 : ℝ) 1) → ℕ)
  (ha : ∀ x, IsGreatest {k | ∃ i : Fin k ↪o Fin n, IsZigZag ((↑) ∘ x ∘ i)} (a x))
: (𝔼[(↑) ∘ a] = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
