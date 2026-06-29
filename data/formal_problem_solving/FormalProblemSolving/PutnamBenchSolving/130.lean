import FormalProblemSolving.Basic
open Filter Topology Nat

/-
## Problem
Let $T_0=2,T_1=3,T_2=6$, and for $n \geq 3$, $T_n=(n+4)T_{n-1}-4nT_{n-2}+(4n-8)T_{n-3}$. The first few terms are $2,3,6,14,40,152,784,5168,40576$. Find, with proof, a formula for $T_n$ of the form $T_n=A_n+B_n$, where $\{A_n\}$ and $\{B_n\}$ are well-known sequences.

## Answer
fun n : ℕ => (n)! + 2 ^ n
-/

-- # Formal Answer
noncomputable abbrev Q130.gt := fun (answer : ℕ → ℤ) => answer = fun (n : ℕ) => ((↑(n ! : ℕ) : ℤ) + ((2 : ℤ) ^ n : ℤ) : ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q130
find (answer : ℕ → ℤ) s.t.
  (T : ℕ → ℤ)
  (hT012 : T 0 = 2 ∧ T 1 = 3 ∧ T 2 = 6)
  (hTn : ∀ n, T (n + 3) = (n + 7) * T (n + 2) - 4 * (n + 3) * T (n + 1) + (4 * n + 4) * T n)
: (T = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q130
find_all (answer : ℕ → ℤ) iff
  (T : ℕ → ℤ)
  (hT012 : T 0 = 2 ∧ T 1 = 3 ∧ T 2 = 6)
  (hTn : ∀ n, T (n + 3) = (n + 7) * T (n + 2) - 4 * (n + 3) * T (n + 1) + (4 * n + 4) * T n)
: (T = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
