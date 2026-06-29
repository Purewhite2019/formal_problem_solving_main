import FormalProblemSolving.Basic
open Function

/-
## Problem
Let $P_n$ be the number of permutations $\pi$ of $\{1,2,\dots,n\}$ such that
\[
|i-j| = 1 \mbox{ implies } |\pi(i) -\pi(j)| \leq 2
\]
for all $i,j$ in $\{1,2,\dots,n\}$. Show that for $n \geq 2$, the quantity
\[
P_{n+5} - P_{n+4} - P_{n+3} + P_n
\]
does not depend on $n$, and find its value.

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q261.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q261
find (answer : ℕ) s.t.
  (P : ℕ → ℕ)
  (hP : P = fun n ↦ {pi : Finset.Icc 1 n → Finset.Icc 1 n | Bijective pi ∧ ∀ i j : Finset.Icc 1 n, Nat.dist i j = 1 → Nat.dist (pi i) (pi j) ≤ 2}.ncard)
: (∀ n : ℕ, n ≥ 2 → (P (n + 5) : ℤ) - (P (n + 4) : ℤ) - (P (n + 3) : ℤ) + (P n : ℤ) = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q261
find_all (answer : ℕ) iff
  (P : ℕ → ℕ)
  (hP : P = fun n ↦ {pi : Finset.Icc 1 n → Finset.Icc 1 n | Bijective pi ∧ ∀ i j : Finset.Icc 1 n, Nat.dist i j = 1 → Nat.dist (pi i) (pi j) ≤ 2}.ncard)
: (∀ n : ℕ, n ≥ 2 → (P (n + 5) : ℤ) - (P (n + 4) : ℤ) - (P (n + 3) : ℤ) + (P n : ℤ) = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
