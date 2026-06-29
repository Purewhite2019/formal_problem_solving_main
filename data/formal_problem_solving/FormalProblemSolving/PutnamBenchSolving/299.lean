import FormalProblemSolving.Basic
open Filter Topology

/-
## Problem
Determine all positive integers $N$ for which the sphere $x^2+y^2+z^2=N$ has an inscribed regular tetrahedron whose vertices have integer coordinates.

## Answer
{3 * m ^ 2 | m > 0}
-/

-- # Formal Answer
noncomputable abbrev Q299.gt := fun (answer : Set ℕ) => answer = ({x : ℕ | ∃ m > (0 : ℕ), ((3 : ℕ) * (m ^ (2 : ℕ) : ℕ) : ℕ) = x} : Set ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q299
find (answer : Set ℕ) s.t.
  (N : ℕ)
  (Nsphere : Set (EuclideanSpace ℝ (Fin 3)))
  (hNsphere : Nsphere = {p | (p 0) ^ 2 + (p 1) ^ 2 + (p 2) ^ 2 = (N : ℝ)})
  (intcoords : (EuclideanSpace ℝ (Fin 3)) → Prop)
  (intcoords_def : ∀ p, intcoords p ↔ ∀ i : Fin 3, p i = round (p i))
: ((0 < N ∧ ∃ A B C D : EuclideanSpace ℝ (Fin 3),
    A ∈ Nsphere ∧ B ∈ Nsphere ∧ C ∈ Nsphere ∧ D ∈ Nsphere ∧
    intcoords A ∧ intcoords B ∧ intcoords C ∧ intcoords D ∧
    (∃ s > 0, dist A B = s ∧ dist A C = s ∧ dist A D = s ∧ dist B C = s ∧ dist B D = s ∧ dist C D = s))
  ↔ N ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q299
find_all (answer : Set ℕ) iff
  (N : ℕ)
  (Nsphere : Set (EuclideanSpace ℝ (Fin 3)))
  (hNsphere : Nsphere = {p | (p 0) ^ 2 + (p 1) ^ 2 + (p 2) ^ 2 = (N : ℝ)})
  (intcoords : (EuclideanSpace ℝ (Fin 3)) → Prop)
  (intcoords_def : ∀ p, intcoords p ↔ ∀ i : Fin 3, p i = round (p i))
: ((0 < N ∧ ∃ A B C D : EuclideanSpace ℝ (Fin 3),
    A ∈ Nsphere ∧ B ∈ Nsphere ∧ C ∈ Nsphere ∧ D ∈ Nsphere ∧
    intcoords A ∧ intcoords B ∧ intcoords C ∧ intcoords D ∧
    (∃ s > 0, dist A B = s ∧ dist A C = s ∧ dist A D = s ∧ dist B C = s ∧ dist B D = s ∧ dist C D = s))
  ↔ N ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
