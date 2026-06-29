import FormalProblemSolving.Basic
open Set MvPolynomial

/-
## Problem
Find all functions $F : \mathbb{R} \setminus \{0, 1\} \to \mathbb{R}$ that satisfy $F(x) + F\left(\frac{x - 1}{x}\right) = 1 + x$ for all $x \in \mathbb{R} \setminus \{0, 1\}$.

## Answer
$\left\{x \mapsto \frac{x^3 - x^2 - 1}{2x(x - 1)}\right\}$
-/

-- # Formal Answer
noncomputable abbrev Q22.gt := fun (answer : Set (ℝ → ℝ)) => answer =
  ({fun (x : ℝ) =>
      ((((x ^ (3 : ℕ) : ℝ) - (x ^ (2 : ℕ) : ℝ) : ℝ) - (1 : ℝ) : ℝ) / (((2 : ℝ) * x : ℝ) * (x - (1 : ℝ) : ℝ) : ℝ) : ℝ)} :
    Set (ℝ → ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q22
find (answer : Set (ℝ → ℝ)) s.t.
  (S : Set ℝ)
  (hS : S = univ \ {0, 1})
  (P : (ℝ → ℝ) → Prop)
  (hP : P = fun (F : ℝ → ℝ) => ∀ x ∈ S, F x + F ((x - 1)/x) = 1 + x)
: (∀ F ∈ answer, P F ∧ ∀ f : ℝ → ℝ, P f → ∃ F ∈ answer, (∀ x ∈ S, f x = F x))
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q22
find_all (answer : Set (ℝ → ℝ)) iff
  (S : Set ℝ)
  (hS : S = univ \ {0, 1})
  (P : (ℝ → ℝ) → Prop)
  (hP : P = fun (F : ℝ → ℝ) => ∀ x ∈ S, F x + F ((x - 1)/x) = 1 + x)
: (∀ F ∈ answer, P F ∧ ∀ f : ℝ → ℝ, P f → ∃ F ∈ answer, (∀ x ∈ S, f x = F x))
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
