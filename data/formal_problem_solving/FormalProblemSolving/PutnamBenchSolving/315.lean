import FormalProblemSolving.Basic
open Nat

/-
## Problem
Let $n$ be an even positive integer. Let $p$ be a monic, real polynomial of degree $2n$; that is to say, $p(x) = x^{2n} + a_{2n-1} x^{2n-1} + \cdots + a_1 x + a_0$ for some real coefficients $a_0, \dots, a_{2n-1}$. Suppose that $p(1/k) = k^2$ for all integers $k$ such that $1 \leq |k| \leq n$. Find all other real numbers $x$ for which $p(1/x) = x^2$.

## Answer
{1/n!, -1/n!}
-/

-- # Formal Answer
noncomputable abbrev Q315.gt := fun (n : ℕ) (answer : ℕ → Set ℝ) => answer = fun (n : ℕ) =>
  (Insert.insert ((1 : ℝ) / (↑(n ! : ℕ) : ℝ) : ℝ) ({((-1 : ℝ) / (↑(n ! : ℕ) : ℝ) : ℝ)} : Set ℝ) : Set ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q315
  (n : ℕ)
find (answer : ℕ → Set ℝ) s.t.
  (hn : n > 0 ∧ Even n)
  (p : Polynomial ℝ)
  (hp : Polynomial.Monic p ∧ p.degree = 2*n)
  (S : Set ℝ)
  (hS : S = {x : ℝ | ∃ k : ℤ, x = k ∧ 1 ≤ |k| ∧ |k| ≤ n})
  (hpinv : ∀ k ∈ S, p.eval (1/k) = k^2)
: ({x : ℝ | x ≠ 0 ∧ p.eval (1/x) = x^2} \ S = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q315
  (n : ℕ)
find_all (answer : ℕ → Set ℝ) iff
  (hn : n > 0 ∧ Even n)
  (p : Polynomial ℝ)
  (hp : Polynomial.Monic p ∧ p.degree = 2*n)
  (S : Set ℝ)
  (hS : S = {x : ℝ | ∃ k : ℤ, x = k ∧ 1 ≤ |k| ∧ |k| ≤ n})
  (hpinv : ∀ k ∈ S, p.eval (1/k) = k^2)
: ({x : ℝ | x ≠ 0 ∧ p.eval (1/x) = x^2} \ S = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
