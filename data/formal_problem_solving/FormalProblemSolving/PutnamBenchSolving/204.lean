import FormalProblemSolving.Basic

/-
## Problem
Let $n$ be a positive odd integer and let $\theta$ be a real number such that $\theta/\pi$ is irrational. Set $a_k=\tan(\theta+k\pi/n)$, $k=1,2,\dots,n$. Prove that $\frac{a_1+a_2+\cdots+a_n}{a_1a_2 \cdots a_n}$ is an integer, and determine its value.

## Answer
(fun n : ℕ => if (n ≡ 1 [MOD 4]) then n else -n)
-/

-- # Formal Answer
noncomputable abbrev Q204.gt := fun (n : ℕ) (answer : ℕ → ℤ) => answer = fun (n : ℕ) => (if n ≡ (1 : ℕ) [MOD (4 : ℕ)] then (↑n : ℤ) else (-(↑n : ℤ) : ℤ) : ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q204
  (n : ℕ)
find (answer : ℕ → ℤ) s.t.
  (theta : ℝ)
  (a : Set.Icc 1 n → ℝ)
  (nodd : Odd n)
  (thetairr : Irrational (theta / Real.pi))
  (ha : ∀ k : Set.Icc 1 n, a k = Real.tan (theta + (k * Real.pi) / n))
: ((∑ k : Set.Icc 1 n, a k) / (∏ k : Set.Icc 1 n, a k) = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q204
  (n : ℕ)
find_all (answer : ℕ → ℤ) iff
  (theta : ℝ)
  (a : Set.Icc 1 n → ℝ)
  (nodd : Odd n)
  (thetairr : Irrational (theta / Real.pi))
  (ha : ∀ k : Set.Icc 1 n, a k = Real.tan (theta + (k * Real.pi) / n))
: ((∑ k : Set.Icc 1 n, a k) / (∏ k : Set.Icc 1 n, a k) = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
