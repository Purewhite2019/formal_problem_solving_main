import FormalProblemSolving.Basic
open Topology Filter

/-
## Problem
Let $n$ be a positive integer. Let $a,b,x$ be real numbers, with $a \neq b$, and let $M_n$ denote the $2n \times 2n$ matrix whose $(i,j)$ entry $m_{ij}$ is given by
\[
m_{ij}=\begin{cases}
x & \text{if }i=j, \\
a & \text{if }i \neq j\text{ and }i+j\text{ is even}, \\
b & \text{if }i \neq j\text{ and }i+j\text{ is odd}.
\end{cases}
\]
Thus, for example, $M_2=\begin{pmatrix} x & b & a & b \\ b & x & b & a \\ a & b & x & b \\ b & a & b & x \end{pmatrix}$. Express $\lim_{x \to a} \det M_n/(x-a)^{2n-2}$ as a polynomial in $a$, $b$, and $n$, where $\det M_n$ denotes the determinant of $M_n$.

## Answer
$(X_2)^2 \cdot ((X_0)^2 - (X_1)^2)$
-/

-- # Formal Answer
noncomputable abbrev Q86.gt := fun (answer : MvPolynomial (Fin 3) ℝ) => answer =
  (((MvPolynomial.X (2 : Fin (3 : ℕ)) : MvPolynomial (Fin (3 : ℕ)) ℝ) ^ (2 : ℕ) : MvPolynomial (Fin (3 : ℕ)) ℝ) *
      (((MvPolynomial.X (0 : Fin (3 : ℕ)) : MvPolynomial (Fin (3 : ℕ)) ℝ) ^ (2 : ℕ) : MvPolynomial (Fin (3 : ℕ)) ℝ) -
          ((MvPolynomial.X (1 : Fin (3 : ℕ)) : MvPolynomial (Fin (3 : ℕ)) ℝ) ^ (2 : ℕ) : MvPolynomial (Fin (3 : ℕ)) ℝ) :
        MvPolynomial (Fin (3 : ℕ)) ℝ) :
    MvPolynomial (Fin (3 : ℕ)) ℝ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q86
find (answer : MvPolynomial (Fin 3) ℝ) s.t.
  (n : ℕ)
  (a : ℝ)
  (b : ℝ)
  (Mn : ℝ → Matrix (Fin (2 * n)) (Fin (2 * n)) ℝ)
  (polyabn : Fin 3 → ℝ)
  (npos : n > 0)
  (aneb : a ≠ b)
  (hMn : Mn = fun x : ℝ => fun i j : Fin (2 * n) => if i = j then x else if Even (i.1 + j.1) then a else b)
  (hpolyabn : polyabn 0 = a ∧ polyabn 1 = b ∧ polyabn 2 = n)
: (Tendsto (fun x : ℝ => (Mn x).det / (x - a) ^ (2 * n - 2)) (𝓝[≠] a) (𝓝 (MvPolynomial.eval polyabn answer)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q86
find_all (answer : MvPolynomial (Fin 3) ℝ) iff
  (n : ℕ)
  (a : ℝ)
  (b : ℝ)
  (Mn : ℝ → Matrix (Fin (2 * n)) (Fin (2 * n)) ℝ)
  (polyabn : Fin 3 → ℝ)
  (npos : n > 0)
  (aneb : a ≠ b)
  (hMn : Mn = fun x : ℝ => fun i j : Fin (2 * n) => if i = j then x else if Even (i.1 + j.1) then a else b)
  (hpolyabn : polyabn 0 = a ∧ polyabn 1 = b ∧ polyabn 2 = n)
: (Tendsto (fun x : ℝ => (Mn x).det / (x - a) ^ (2 * n - 2)) (𝓝[≠] a) (𝓝 (MvPolynomial.eval polyabn answer)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
