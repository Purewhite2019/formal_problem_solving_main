import FormalProblemSolving.Basic
open EuclideanGeometry Filter Topology Set

/-
## Problem
We call a function $f$ from $[0,1]$ to the reals to be supercontinuous on $[0,1]$ if the Cesaro-limit exists for the sequence $f(x_1), f(x_2), f(x_3), \dots$ whenever it does for the sequence $x_1, x_2, x_3 \dots$. Find all supercontinuous functions on $[0,1]$.

## Answer
the set of all linear functions on [0,1]
-/

-- # Formal Answer
noncomputable abbrev Q23.gt := fun (answer : Set (ℝ → ℝ)) => answer =
  ({f : ℝ → ℝ |
      ∃ (A : ℝ) (B : ℝ), ∀ (x : ℝ), x ∈ (Set.Icc (0 : ℝ) (1 : ℝ) : Set ℝ) → (f x : ℝ) = ((A * x : ℝ) + B : ℝ)} :
    Set (ℝ → ℝ))

-- # Formal Problem (FPS)
namespace PutnamBenchSolving_FPS
open scoped FPS

problem Q23
find (answer : Set (ℝ → ℝ)) s.t.
  (climit_exists : (ℕ → ℝ) → Prop)
  (supercontinuous : (ℝ → ℝ) → Prop)
  (hclimit_exists : ∀ x, climit_exists x ↔ ∃ C : ℝ, Tendsto (fun n => (∑ i in Finset.range n, (x i))/(n : ℝ)) atTop (𝓝 C))
  (hsupercontinuous : ∀ f, supercontinuous f ↔ ∀ (x : ℕ → ℝ), (∀ i : ℕ, x i ∈ Icc 0 1) → climit_exists x → climit_exists (fun i => f (x i)))
: ({f | supercontinuous f} = answer)
:= solve
-- Placeholder
all_goals sorry

end PutnamBenchSolving_FPS

-- # Formal Problem (D-FPS)
namespace PutnamBenchSolving_DFPS
open scoped DFPS

problem Q23
find_all (answer : Set (ℝ → ℝ)) iff
  (climit_exists : (ℕ → ℝ) → Prop)
  (supercontinuous : (ℝ → ℝ) → Prop)
  (hclimit_exists : ∀ x, climit_exists x ↔ ∃ C : ℝ, Tendsto (fun n => (∑ i in Finset.range n, (x i))/(n : ℝ)) atTop (𝓝 C))
  (hsupercontinuous : ∀ f, supercontinuous f ↔ ∀ (x : ℕ → ℝ), (∀ i : ℕ, x i ∈ Icc 0 1) → climit_exists x → climit_exists (fun i => f (x i)))
: ({f | supercontinuous f} = answer)
:= solve
case Answer := sorry
all_goals sorry

end PutnamBenchSolving_DFPS
