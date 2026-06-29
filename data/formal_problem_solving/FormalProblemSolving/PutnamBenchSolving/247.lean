import FormalProblemSolving.Basic
open Function Set

/-
## Problem
For positive integers $n$, let the numbers $c(n)$ be determined by the rules $c(1)=1$, $c(2n)=c(n)$, and $c(2n+1)=(-1)^nc(n)$. Find the value of $\sum_{n=1}^{2013} c(n)c(n+2)$.

## Answer
-1
-/

-- # Formal Answer
noncomputable abbrev Q247.gt := fun (answer : ℤ) => answer = (-1 : ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q247
find (answer : ℤ) s.t.
  (c : ℕ → ℤ)
  (hc1 : c 1 = 1)
  (hceven : ∀ n : ℕ, n > 0 → c (2 * n) = c n)
  (hcodd : ∀ n : ℕ, n > 0 → c (2 * n + 1) = (-1) ^ n * c n)
: ((∑ n : Set.Icc 1 2013, c n * c (n.1 + 2)) = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q247
find_all (answer : ℤ) iff
  (c : ℕ → ℤ)
  (hc1 : c 1 = 1)
  (hceven : ∀ n : ℕ, n > 0 → c (2 * n) = c n)
  (hcodd : ∀ n : ℕ, n > 0 → c (2 * n + 1) = (-1) ^ n * c n)
: ((∑ n : Set.Icc 1 2013, c n * c (n.1 + 2)) = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
