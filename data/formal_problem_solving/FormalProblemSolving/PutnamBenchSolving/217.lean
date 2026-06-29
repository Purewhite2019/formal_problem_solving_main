import FormalProblemSolving.Basic
open Filter Topology Set

/-
## Problem
What is the maximum number of rational points that can lie on a circle in $\mathbb{R}^2$ whose center is not a rational point? (A \emph{rational point} is a point both of whose coordinates are rational numbers.)

## Answer
2
-/

-- # Formal Answer
noncomputable abbrev Q217.gt := fun (answer : ℕ) => answer = (2 : ℕ)

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q217
find (answer : ℕ) s.t.
  (is_rational_point : EuclideanSpace ℝ (Fin 2) → Prop)
  (h_rational_point : ∀ p : EuclideanSpace ℝ (Fin 2), is_rational_point p ↔ ∃ (a b : ℚ), a = p 0 ∧ b = p 1)
  (real_circle : EuclideanSpace ℝ (Fin 2) → ℝ → Set (EuclideanSpace ℝ (Fin 2)))
  (h_real_circle : ∀ (c : EuclideanSpace ℝ (Fin 2)) (r : ℝ), real_circle c r = {p | dist p c = r})
: (IsGreatest {n : ℕ | ∃ (c : EuclideanSpace ℝ (Fin 2)) (r : ℝ), ¬ is_rational_point c ∧ (Set.ncard {p : EuclideanSpace ℝ (Fin 2) | p ∈ real_circle c r ∧ is_rational_point p} = n)} answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q217
find_all (answer : ℕ) iff
  (is_rational_point : EuclideanSpace ℝ (Fin 2) → Prop)
  (h_rational_point : ∀ p : EuclideanSpace ℝ (Fin 2), is_rational_point p ↔ ∃ (a b : ℚ), a = p 0 ∧ b = p 1)
  (real_circle : EuclideanSpace ℝ (Fin 2) → ℝ → Set (EuclideanSpace ℝ (Fin 2)))
  (h_real_circle : ∀ (c : EuclideanSpace ℝ (Fin 2)) (r : ℝ), real_circle c r = {p | dist p c = r})
: (IsGreatest {n : ℕ | ∃ (c : EuclideanSpace ℝ (Fin 2)) (r : ℝ), ¬ is_rational_point c ∧ (Set.ncard {p : EuclideanSpace ℝ (Fin 2) | p ∈ real_circle c r ∧ is_rational_point p} = n)} answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
