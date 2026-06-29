import FormalProblemSolving.Basic
open Set Function Filter Topology Polynomial Real

/-
## Problem
Let $n_1, n_2, \dots, n_s$ be distinct integers such that, for every integer $k$, $n_1n_2\cdots n_s$ divides $(n_1 + k)(n_2 + k) \cdots (n_s + k)$. Prove or provide a counterexample to the following claims:
\begin{enumerate}
\item
For some $i$, $|n_i| = 1$.
\item
If all $n_i$ are positive, then $\{n_1, n_2, \dots, n_s\} = \{1, 2, \dots, s\}$.
\end{enumerate}

## Answer
(True, True)
-/

-- # Formal Answer
noncomputable abbrev Q79.gt := fun (answer : Prop × Prop) => answer = ((True, True) : Prop × Prop)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q79
find (answer : Prop × Prop) s.t.
  (P : Finset ℤ → Prop)
  (P_def : ∀ n, P n ↔ n.Nonempty ∧ ∀ k, ∏ i in n, i ∣ ∏ i in n, (i + k))
: (((∀ n, P n → 1 ∈ n ∨ -1 ∈ n) ↔ answer.1) ∧
    ((∀ n, P n → (∀ i ∈ n, 0 < i) → n = Finset.Icc (1 : ℤ) n.card) ↔ answer.2))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q79
find_all (answer : Prop × Prop) iff
  (P : Finset ℤ → Prop)
  (P_def : ∀ n, P n ↔ n.Nonempty ∧ ∀ k, ∏ i in n, i ∣ ∏ i in n, (i + k))
: (((∀ n, P n → 1 ∈ n ∨ -1 ∈ n) ↔ answer.1) ∧
    ((∀ n, P n → (∀ i ∈ n, 0 < i) → n = Finset.Icc (1 : ℤ) n.card) ↔ answer.2))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
