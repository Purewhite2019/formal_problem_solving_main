import FormalProblemSolving.Basic
open Real Equiv

/-
## Problem
Let $a_1, a_2, \dots, a_n$ be real numbers, and let $b_1, b_2, \dots, b_n$ be distinct positive integers. Suppose that there is a polynomial $f(x)$ satisfying the identity
\[
(1-x)^n f(x) = 1 + \sum_{i=1}^n a_i x^{b_i}.
\]
Find a simple expression (not involving any sums) for $f(1)$ in terms of $b_1, b_2, \dots, b_n$ and $n$ (but independent of $a_1, a_2, \dots, a_n$).

## Answer
fun b n ↦ (∏ i : Finset.Icc 1 n, b i) / Nat.factorial n
-/

-- # Formal Answer
noncomputable abbrev Q105.gt := fun (n : ℕ) (b : ℕ → ℕ) (answer : (ℕ → ℕ) → ℕ → ℝ) => answer = fun (b : ℕ → ℕ) (n : ℕ) =>
  ((↑(∏ i : { x : ℕ // x ∈ (Finset.Icc (1 : ℕ) n : Finset ℕ) }, (b (↑i : ℕ) : ℕ)) : ℝ) / (↑(n.factorial : ℕ) : ℝ) : ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q105
  (n : ℕ)
  (b : ℕ → ℕ)
find (answer : (ℕ → ℕ) → ℕ → ℝ) s.t.
  (npos : n > 0)
  (a : ℕ → ℝ)
  (bpos : ∀ i ∈ Finset.Icc 1 n, b i > 0)
  (binj : ∀ i ∈ Finset.Icc 1 n, ∀ j ∈ Finset.Icc 1 n, b i = b j → i = j)
  (f : Polynomial ℝ)
  (hf : ∀ x : ℝ, (1 - x) ^ n * f.eval x = 1 + ∑ i : Finset.Icc 1 n, (a i) * x ^ (b i))
: (f.eval 1 = answer b n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q105
  (n : ℕ)
  (b : ℕ → ℕ)
find_all (answer : (ℕ → ℕ) → ℕ → ℝ) iff
  (npos : n > 0)
  (a : ℕ → ℝ)
  (bpos : ∀ i ∈ Finset.Icc 1 n, b i > 0)
  (binj : ∀ i ∈ Finset.Icc 1 n, ∀ j ∈ Finset.Icc 1 n, b i = b j → i = j)
  (f : Polynomial ℝ)
  (hf : ∀ x : ℝ, (1 - x) ^ n * f.eval x = 1 + ∑ i : Finset.Icc 1 n, (a i) * x ^ (b i))
: (f.eval 1 = answer b n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
