import FormalProblemSolving.Basic
open Set Filter Topology

/-
## Problem
For positive integers $n$, let $M_n$ be the $2n+1$ by $2n+1$ skew-symmetric matrix for which each entry in the first $n$ subdiagonals below the main diagonal is $1$ and each of the remaining entries below the main diagonal is $-1$. Find, with proof, the rank of $M_n$. (According to one definition, the rank of a matrix is the largest $k$ such that there is a $k \times k$ submatrix with nonzero determinant.) One may note that
\begin{align*}
M_1&=\begin{pmatrix} 0 & -1 & 1 \\ 1 & 0 & -1 \\ -1 & 1 & 0 \end{pmatrix} \\
M_2&=\begin{pmatrix} 0 & -1 & -1 & 1 & 1 \\ 1 & 0 & -1 & -1 & 1 \\ 1 & 1 & 0 & -1 & -1 \\ -1 & 1 & 1 & 0 & -1 \\ -1 & -1 & 1 & 1 & 0 \end{pmatrix}.
\end{align*}

## Answer
2n
-/

-- # Formal Answer
noncomputable abbrev Q123.gt := fun (n : ℕ) (answer : ℕ → ℕ) => answer = fun (n : ℕ) => ((2 : ℕ) * n : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q123
  (n : ℕ)
find (answer : ℕ → ℕ) s.t.
  (hn : n > 0)
  (Mn : Matrix (Fin (2 * n + 1)) (Fin (2 * n + 1)) ℝ)
  (Mnskewsymm : ∀ i j, Mn i j = -(Mn j i))
  (hMn1 : ∀ i j, (1 ≤ (i.1 : ℤ) - j.1 ∧ (i.1 : ℤ) - j.1 ≤ n) → Mn i j = 1)
  (hMnn1 : ∀ i j, (i.1 : ℤ) - j.1 > n → Mn i j = -1)
: (Mn.rank = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q123
  (n : ℕ)
find_all (answer : ℕ → ℕ) iff
  (hn : n > 0)
  (Mn : Matrix (Fin (2 * n + 1)) (Fin (2 * n + 1)) ℝ)
  (Mnskewsymm : ∀ i j, Mn i j = -(Mn j i))
  (hMn1 : ∀ i j, (1 ≤ (i.1 : ℤ) - j.1 ∧ (i.1 : ℤ) - j.1 ≤ n) → Mn i j = 1)
  (hMnn1 : ∀ i j, (i.1 : ℤ) - j.1 > n → Mn i j = -1)
: (Mn.rank = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
