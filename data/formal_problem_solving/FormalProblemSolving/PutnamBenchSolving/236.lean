import FormalProblemSolving.Basic

/-
## Problem
Define a \emph{growing spiral} in the plane to be a sequence of points with integer coordinates $P_0=(0,0),P_1,\dots,P_n$ such that $n \geq 2$ and:
\begin{itemize}
\item the directed line segments $P_0P_1,P_1P_2,\dots,P_{n-1}P_n$ are in the successive coordinate directions east (for $P_0P_1$), north, west, south, east, etc.;
\item the lengths of these line segments are positive and strictly increasing.
\end{itemize}
How many of the points $(x,y)$ with integer coordinates $0 \leq x \leq 2011,0 \leq y \leq 2011$ \emph{cannot} be the last point, $P_n$ of any growing spiral?

## Answer
10053
-/

-- # Formal Answer
noncomputable abbrev Q236.gt := fun (answer : ℕ) => answer = (10053 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q236
find (answer : ℕ) s.t.
  (IsSpiral : List (Fin 2 → ℤ) → Prop)
  (IsSpiral_def : ∀ P, IsSpiral P ↔ P.length ≥ 3 ∧ P[0]! = 0 ∧
  (∃ l : Fin (P.length - 1) → ℕ, l > 0 ∧ StrictMono l ∧ (∀ i : Fin (P.length - 1),
    (i.1 % 4 = 0 → (P[i] 0 + l i = P[i.1 + 1]! 0 ∧ P[i] 1 = P[i.1 + 1]! 1)) ∧
    (i.1 % 4 = 1 → (P[i] 0 = P[i.1 + 1]! 0 ∧ P[i] 1 + l i = P[i.1 + 1]! 1)) ∧
    (i.1 % 4 = 2 → (P[i] 0 - l i = P[i.1 + 1]! 0 ∧ P[i] 1 = P[i.1 + 1]! 1)) ∧
    (i.1 % 4 = 3 → (P[i] 0 = P[i.1 + 1]! 0 ∧ P[i] 1 - l i = P[i.1 + 1]! 1)))))
: ({p | 0 ≤ p 0 ∧ p 0 ≤ 2011 ∧ 0 ≤ p 1 ∧ p 1 ≤ 2011 ∧ ¬∃ spiral, IsSpiral spiral ∧ spiral.getLast! = p}.encard = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q236
find_all (answer : ℕ) iff
  (IsSpiral : List (Fin 2 → ℤ) → Prop)
  (IsSpiral_def : ∀ P, IsSpiral P ↔ P.length ≥ 3 ∧ P[0]! = 0 ∧
  (∃ l : Fin (P.length - 1) → ℕ, l > 0 ∧ StrictMono l ∧ (∀ i : Fin (P.length - 1),
    (i.1 % 4 = 0 → (P[i] 0 + l i = P[i.1 + 1]! 0 ∧ P[i] 1 = P[i.1 + 1]! 1)) ∧
    (i.1 % 4 = 1 → (P[i] 0 = P[i.1 + 1]! 0 ∧ P[i] 1 + l i = P[i.1 + 1]! 1)) ∧
    (i.1 % 4 = 2 → (P[i] 0 - l i = P[i.1 + 1]! 0 ∧ P[i] 1 = P[i.1 + 1]! 1)) ∧
    (i.1 % 4 = 3 → (P[i] 0 = P[i.1 + 1]! 0 ∧ P[i] 1 - l i = P[i.1 + 1]! 1)))))
: ({p | 0 ≤ p 0 ∧ p 0 ≤ 2011 ∧ 0 ≤ p 1 ∧ p 1 ≤ 2011 ∧ ¬∃ spiral, IsSpiral spiral ∧ spiral.getLast! = p}.encard = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
