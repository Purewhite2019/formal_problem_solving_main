import FormalProblemSolving.Basic
open Filter Topology Set Nat

/-
## Problem
Find all continuously differentiable functions f : \mathbb{R} \to \mathbb{R} such that for every rational number $q$, the number $f(q)$ is rational and has the same denominator as $q$.

## Answer
{fun (x : ℝ) => x + n | n : ℤ} ∪ {fun (x : ℝ) => -x + n | n : ℤ}
-/

-- # Formal Answer
noncomputable abbrev Q220.gt := fun (answer : Set (ℝ → ℝ)) => answer =
  (({x : ℝ → ℝ | ∃ (n : ℤ), (fun (x : ℝ) => (x + (↑n : ℝ) : ℝ)) = x} : Set (ℝ → ℝ)) ∪
      ({x : ℝ → ℝ | ∃ (n : ℤ), (fun (x : ℝ) => ((-x : ℝ) + (↑n : ℝ) : ℝ)) = x} : Set (ℝ → ℝ)) :
    Set (ℝ → ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q220
find (answer : Set (ℝ → ℝ)) s.t.
  (fqsat : (ℝ → ℝ) → ℚ → Prop)
  (hfqsat : ∀ f q, fqsat f q ↔ ContDiff ℝ 1 f ∧ (∃ p : ℚ, p = f q ∧ p.den = q.den))
: (∀ f : (ℝ → ℝ), (∀ q : ℚ, fqsat f q) ↔ f ∈ answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q220
find_all (answer : Set (ℝ → ℝ)) iff
  (fqsat : (ℝ → ℝ) → ℚ → Prop)
  (hfqsat : ∀ f q, fqsat f q ↔ ContDiff ℝ 1 f ∧ (∃ p : ℚ, p = f q ∧ p.den = q.den))
: (∀ f : (ℝ → ℝ), (∀ q : ℚ, fqsat f q) ↔ f ∈ answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
