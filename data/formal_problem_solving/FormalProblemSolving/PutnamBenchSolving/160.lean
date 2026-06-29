import FormalProblemSolving.Basic
open Filter Topology Real Nat

/-
## Problem
To each positive integer with $n^{2}$ decimal digits, we associate the determinant of the matrix obtained by writing the digits in order across the rows. For example, for $n=2$, to the integer 8617 we associate $\det \left( \begin{array}{cc} 8 & 6 \ 1 & 7 \end{array} \right) = 50$. Find, as a function of $n$, the sum of all the determinants associated with $n^{2}$-digit integers. (Leading digits are assumed to be nonzero; for example, for $n=2$, there are 9000 determinants.)

## Answer
45 if n = 1, 10 * 45² if n = 2, and 0 otherwise
-/

-- # Formal Answer
noncomputable abbrev Q160.gt := fun (n : ℕ) (answer : ℕ → ℤ) => answer = fun (n : ℕ) =>
  (if n = (1 : ℕ) then (45 : ℤ) else (if n = (2 : ℕ) then ((10 : ℤ) * ((45 : ℤ) ^ (2 : ℕ) : ℤ) : ℤ) else (0 : ℤ) : ℤ) :
    ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q160
  (n : ℕ)
find (answer : ℕ → ℤ) s.t.
  (hn : n > 0)
  (digits_set : Set (ℕ → ℕ))
  (hdigits_set : digits_set = {f | f 0 ≠ 0 ∧ (∀ i : Fin (n ^ 2), f i ≤ 9) ∧ (∀ i ≥ n ^ 2, f i = 0)})
  (digits_to_matrix : (ℕ → ℕ) → Matrix (Fin n) (Fin n) ℤ)
  (hdigits_to_matrix : digits_to_matrix = fun f => (fun i j => f (i.1 * n + j.1)))
: (∑' f : digits_set, (digits_to_matrix f).det = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q160
  (n : ℕ)
find_all (answer : ℕ → ℤ) iff
  (hn : n > 0)
  (digits_set : Set (ℕ → ℕ))
  (hdigits_set : digits_set = {f | f 0 ≠ 0 ∧ (∀ i : Fin (n ^ 2), f i ≤ 9) ∧ (∀ i ≥ n ^ 2, f i = 0)})
  (digits_to_matrix : (ℕ → ℕ) → Matrix (Fin n) (Fin n) ℤ)
  (hdigits_to_matrix : digits_to_matrix = fun f => (fun i j => f (i.1 * n + j.1)))
: (∑' f : digits_set, (digits_to_matrix f).det = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
