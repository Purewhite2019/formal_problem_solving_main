import FormalProblemSolving.Basic
open Set

/-
## Problem
Determine, with proof, the number of ordered triples $(A_1, A_2, A_3)$ of sets which have the property that
\begin{enumerate}
\item[(i)] $A_1 \cup A_2 \cup A_3 = \{1,2,3,4,5,6,7,8,9,10\}$, and
\item[(ii)] $A_1 \cap A_2 \cap A_3 = \emptyset$.
\end{enumerate}
Express your answer in the form $2^a 3^b 5^c 7^d$, where $a,b,c,d$ are nonnegative integers.

## Answer
(10, 10, 0, 0)
-/

-- # Formal Answer
noncomputable abbrev Q93.gt := fun (answer : ℕ × ℕ × ℕ × ℕ) => answer = (((10 : ℕ), (((10 : ℕ), (((0 : ℕ), (0 : ℕ)) : ℕ × ℕ)) : ℕ × ℕ × ℕ)) : ℕ × ℕ × ℕ × ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q93
find (answer : ℕ × ℕ × ℕ × ℕ) s.t.
: (let (a, b, c, d) := answer;
  {(A1, A2, A3) : Set ℤ × Set ℤ × Set ℤ | A1 ∪ A2 ∪ A3 = Icc 1 10 ∧ A1 ∩ A2 ∩ A3 = ∅}.ncard = 2 ^ a * 3 ^ b * 5 ^ c * 7 ^ d)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q93
find_all (answer : ℕ × ℕ × ℕ × ℕ) iff
: (let (a, b, c, d) := answer;
  {(A1, A2, A3) : Set ℤ × Set ℤ × Set ℤ | A1 ∪ A2 ∪ A3 = Icc 1 10 ∧ A1 ∩ A2 ∩ A3 = ∅}.ncard = 2 ^ a * 3 ^ b * 5 ^ c * 7 ^ d)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
