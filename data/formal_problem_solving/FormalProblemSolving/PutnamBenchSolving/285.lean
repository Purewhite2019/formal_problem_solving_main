import FormalProblemSolving.Basic
open Topology Filter

/-
## Problem
Denote by $\mathbb{Z}^2$ the set of all points $(x,y)$ in the plane with integer coordinates. For each integer $n \geq 0$, let $P_n$ be the subset of $\mathbb{Z}^2$ consisting of the point $(0,0)$ together with all points $(x,y)$ such that $x^2+y^2=2^k$ for some integer $k \leq n$. Determine, as a function of $n$, the number of four-point subsets of $P_n$ whose elements are the vertices of a square.

## Answer
5n + 1
-/

-- # Formal Answer
noncomputable abbrev Q285.gt := fun (n : ℕ) (answer : ℕ → ℕ) => answer = fun (n : ℕ) => (((5 : ℕ) * n : ℕ) + (1 : ℕ) : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q285
  (n : ℕ)
find (answer : ℕ → ℕ) s.t.
  (Pn : Set (Fin 2 → ℤ))
  (pZtoR : (Fin 2 → ℤ) → EuclideanSpace ℝ (Fin 2))
  (sPnsquare : Finset (Fin 2 → ℤ) → Prop)
  (hPn : Pn = {p | (p 0 = 0 ∧ p 1 = 0) ∨ (∃ k ≤ n, (p 0) ^ 2 + (p 1) ^ 2 = 2 ^ k)})
  (hpZtoR : ∀ p i, (pZtoR p) i = p i)
  (sPnsquare_def : ∀ sPn : Finset (Fin 2 → ℤ), sPnsquare sPn ↔ (sPn.card = 4 ∧ ∃ p4 : Fin 4 → (Fin 2 → ℤ), Set.range p4 = sPn ∧ (∃ s > 0, ∀ i : Fin 4, dist (pZtoR (p4 i) : EuclideanSpace ℝ (Fin 2)) (pZtoR (p4 (i + 1))) = s) ∧ (dist (pZtoR (p4 0)) (pZtoR (p4 2)) = dist (pZtoR (p4 1)) (pZtoR (p4 3)))))
: ({sPn : Finset (Fin 2 → ℤ) | (sPn : Set (Fin 2 → ℤ)) ⊆ Pn ∧ sPnsquare sPn}.encard = answer n)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q285
  (n : ℕ)
find_all (answer : ℕ → ℕ) iff
  (Pn : Set (Fin 2 → ℤ))
  (pZtoR : (Fin 2 → ℤ) → EuclideanSpace ℝ (Fin 2))
  (sPnsquare : Finset (Fin 2 → ℤ) → Prop)
  (hPn : Pn = {p | (p 0 = 0 ∧ p 1 = 0) ∨ (∃ k ≤ n, (p 0) ^ 2 + (p 1) ^ 2 = 2 ^ k)})
  (hpZtoR : ∀ p i, (pZtoR p) i = p i)
  (sPnsquare_def : ∀ sPn : Finset (Fin 2 → ℤ), sPnsquare sPn ↔ (sPn.card = 4 ∧ ∃ p4 : Fin 4 → (Fin 2 → ℤ), Set.range p4 = sPn ∧ (∃ s > 0, ∀ i : Fin 4, dist (pZtoR (p4 i) : EuclideanSpace ℝ (Fin 2)) (pZtoR (p4 (i + 1))) = s) ∧ (dist (pZtoR (p4 0)) (pZtoR (p4 2)) = dist (pZtoR (p4 1)) (pZtoR (p4 3)))))
: ({sPn : Finset (Fin 2 → ℤ) | (sPn : Set (Fin 2 → ℤ)) ⊆ Pn ∧ sPnsquare sPn}.encard = answer n)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
