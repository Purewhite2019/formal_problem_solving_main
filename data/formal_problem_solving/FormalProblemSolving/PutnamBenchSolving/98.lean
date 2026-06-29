import FormalProblemSolving.Basic
open Set Filter Topology Real Polynomial Function

/-
## Problem
Let $k$ be the smallest positive integer for which there exist distinct integers $m_1, m_2, m_3, m_4, m_5$ such that the polynomial
\[
p(x) = (x-m_1)(x-m_2)(x-m_3)(x-m_4)(x-m_5)
\]
has exactly $k$ nonzero coefficients. Find, with proof, a set of integers $m_1, m_2, m_3, m_4, m_5$ for which this minimum $k$ is achieved.

## Answer
fun i : Fin 5 ↦ ↑i - (2 : ℤ)
-/

-- # Formal Answer
noncomputable abbrev Q98.gt := fun (answer : Fin 5 → ℤ) => answer = fun (i : Fin (5 : ℕ)) => ((↑(↑i : ℕ) : ℤ) - (2 : ℤ) : ℤ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q98
find (answer : Fin 5 → ℤ) s.t.
  (p : (Fin 5 → ℤ) → (Polynomial ℝ))
  (hp : p = fun m ↦ ∏ i : Fin 5, ((X : Polynomial ℝ) - m i))
  (numnzcoeff : Polynomial ℝ → ℕ)
  (hnumnzcoeff : numnzcoeff = fun p ↦ {j ∈ Finset.range (p.natDegree + 1) | coeff p j ≠ 0}.card)
: ((Injective answer ∧ ∀ m : Fin 5 → ℤ, Injective m → numnzcoeff (p answer) ≤ numnzcoeff (p m)))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q98
find_all (answer : Fin 5 → ℤ) iff
  (p : (Fin 5 → ℤ) → (Polynomial ℝ))
  (hp : p = fun m ↦ ∏ i : Fin 5, ((X : Polynomial ℝ) - m i))
  (numnzcoeff : Polynomial ℝ → ℕ)
  (hnumnzcoeff : numnzcoeff = fun p ↦ {j ∈ Finset.range (p.natDegree + 1) | coeff p j ≠ 0}.card)
: ((Injective answer ∧ ∀ m : Fin 5 → ℤ, Injective m → numnzcoeff (p answer) ≤ numnzcoeff (p m)))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
