import FormalProblemSolving.Basic
open Topology Filter Function Nat

/-
## Problem
Let $n$ be a positive integer, and let $f(n)$ denote the last nonzero digit in the decimal expansion of $n!$. For instance, $f(5)=2$.
\begin{enumerate}
\item[(a)] Show that if $a_1,a_2,\dots,a_k$ are \emph{distinct} nonnegative integers, then $f(5^{a_1}+5^{a_2}+\dots+5^{a_k})$ depends only on the sum $a_1+a_2+\dots+a_k$.
\item[(b)] Assuming part (a), we can define $g(s)=f(5^{a_1}+5^{a_2}+\dots+5^{a_k})$, where $s=a_1+a_2+\dots+a_k$. Find the least positive integer $p$ for which $g(s)=g(s + p)$, for all $s \geq 1$, or else show that no such $p$ exists.
\end{enumerate}

## Answer
4
-/

-- # Formal Answer
noncomputable abbrev Q88.gt := fun (answer : ℕ) => answer = (4 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q88
find (answer : ℕ) s.t.
  (f : ℕ → ℕ)
  (hf : ∀ n, some (f n) = (Nat.digits 10 (n !)).find? (fun d ↦ d ≠ 0))
  (IsPeriodicFrom : ℕ → (ℕ → ℕ) → ℕ → Prop)
  (IsPeriodicFrom_def : ∀ x f p, IsPeriodicFrom x f p ↔ Periodic (f ∘ (· + x)) p)
  (P : ℕ → (ℕ → ℕ) → ℕ → Prop)
  (P_def : ∀ x g p, P x g p ↔ if p = 0 then (∀ q > 0, ¬ IsPeriodicFrom x g q) else
        IsLeast {q | 0 < q ∧ IsPeriodicFrom x g q} p)
: (∃ g : ℕ → ℕ,
      (∀ᵉ (k > 0) (a : Fin k → ℕ) (ha : Injective a), f (∑ i, 5 ^ (a i)) = g (∑ i, a i)) ∧
      P 1 g answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q88
find_all (answer : ℕ) iff
  (f : ℕ → ℕ)
  (hf : ∀ n, some (f n) = (Nat.digits 10 (n !)).find? (fun d ↦ d ≠ 0))
  (IsPeriodicFrom : ℕ → (ℕ → ℕ) → ℕ → Prop)
  (IsPeriodicFrom_def : ∀ x f p, IsPeriodicFrom x f p ↔ Periodic (f ∘ (· + x)) p)
  (P : ℕ → (ℕ → ℕ) → ℕ → Prop)
  (P_def : ∀ x g p, P x g p ↔ if p = 0 then (∀ q > 0, ¬ IsPeriodicFrom x g q) else
        IsLeast {q | 0 < q ∧ IsPeriodicFrom x g q} p)
: (∃ g : ℕ → ℕ,
      (∀ᵉ (k > 0) (a : Fin k → ℕ) (ha : Injective a), f (∑ i, 5 ^ (a i)) = g (∑ i, a i)) ∧
      P 1 g answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
